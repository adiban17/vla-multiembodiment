"""The unified training sample, the same for both robots (spec: "One sample, both robots").

One sample = one moment t of one demo episode:

    vision          2 x 65 x 384   cached DINOv2 features of frames t-1 and t   (mode="cached")
    images          2 x 3 x 224 x 224   preprocessed frames t-1 and t           (mode="raw")
    pixels          2 x H x W x 3  uint8 frames t-1 and t, untouched            (mode="pixels")
                                   (baselines: the model runs preprocess_image() itself, on the GPU)
    lang            384            MiniLM vector of the instruction
    instruction     text, and instruction_id (index into language.npy)
    state           8              normalized; PushT has 2 real + 6 zeros
    embodiment_id   scalar         0 = PushT, 1 = FetchPush
    actions         16 x 4         normalized actions t .. t+15; PushT has 2 real + 2 zero dims
    action_dim_mask 4              True for the robot's real action dims
    action_is_pad   16             True for steps past the episode end (those actions are 0)
    frame, episode  ints           where the sample came from (for checks and debugging)

Rules that are easy to get wrong (each has a test):
- At the first step of an episode there is no frame t-1, so frame t is repeated.
- An action chunk never reaches into the next episode: steps past the end are padding.

The data comes from one robot's feature-cache folder (written by vla.data.cache_features):
lowdim.npz, vision.npy (fp16) and images.npy (uint8), both opened memory-mapped, so random access
is fast and nothing is loaded into RAM up front. stats.json and language.npy sit one level up.
"""

from pathlib import Path

import numpy as np
import torch

from vla.data.instructions import N_TRAIN, PARAPHRASES, all_instructions, instruction_id
from vla.data.normalize import Normalizer
from vla.data.shards import assert_training_images_ok, read_meta

N_FRAMES = 2
CHUNK = 16
STATE_DIM = 8
ACTION_DIM = 4
EMBODIMENT_ID = {"pusht": 0, "fetch": 1}
MODES = ("cached", "raw", "pixels")


def split_episodes(n_episodes: int, val_frac: float = 0.1, seed: int = 0) -> tuple[list[int], list[int]]:
    """Split episode indices into (train, val), by episode, the same way every time."""
    n_val = max(1, round(n_episodes * val_frac)) if n_episodes > 1 else 0
    order = np.random.default_rng(seed).permutation(n_episodes)
    return sorted(order[n_val:].tolist()), sorted(order[:n_val].tolist())


def _pad_last(x: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros((*x.shape[:-1], size), dtype=np.float32)
    out[..., :x.shape[-1]] = x
    return out


class WindowDataset(torch.utils.data.Dataset):
    """Samples of one robot. Combine the two robots with a sampler in Phase 3 (50/50 batches).

    robot_dir      <cache run>/<robot>/ from vla.data.cache_features
    split          "train", "val" (the episode split saved in stats.json) or "all"
    mode           "cached" (DINOv2 features, frozen runs), "raw" (preprocessed images, adaptation
                   runs) or "pixels" (uint8 frames, baselines)
    instructions   "train" (random one of 8), "heldout" (random one of 2), or k = fixed paraphrase k
    require_training_images   refuse debug-only images (only tests turn this off)
    """

    def __init__(self, robot_dir, split: str = "train", mode: str = "cached", normalizer: Normalizer | None = None,
                 instructions="train", require_training_images: bool = True, seed: int = 0):
        self.robot_dir = Path(robot_dir)
        self.meta = read_meta(self.robot_dir)
        if self.meta.get("kind") != "features":
            raise ValueError(f"{self.robot_dir} is not a feature-cache folder (see vla.data.cache_features)")
        if require_training_images:
            assert_training_images_ok(self.meta)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.robot = self.meta["robot"]
        self.mode = mode
        root = self.robot_dir.parent
        self.normalizer = normalizer or Normalizer.load(root / "stats.json")

        low = np.load(self.robot_dir / "lowdim.npz")
        lengths = low["episode_length"].astype(np.int64)
        self.episode_ends = np.cumsum(lengths)
        self.episode_starts = self.episode_ends - lengths
        self.episode_of_frame = np.repeat(np.arange(len(lengths)), lengths)

        if split == "all":
            episodes = list(range(len(lengths)))
        elif split in ("train", "val"):
            episodes = self.normalizer.stats["split"][self.robot][split]
        else:
            raise ValueError(f"split must be 'train', 'val' or 'all', got {split!r}")
        self.frames = np.concatenate(
            [np.arange(self.episode_starts[e], self.episode_ends[e]) for e in episodes]
        ) if episodes else np.zeros(0, dtype=np.int64)

        # Normalize once, then pad to the shared sizes (zeros in the padded dims)
        self.state = _pad_last(self.normalizer.normalize(low["state"], self.robot, "state"), STATE_DIM)
        actions = self.normalizer.normalize(low["action"], self.robot, "action")
        self.action_dims = actions.shape[-1]
        self.action = _pad_last(actions, ACTION_DIM)
        self.action_dim_mask = torch.arange(ACTION_DIM) < self.action_dims

        self.language = np.load(root / "language.npy").astype(np.float32)
        robot_ids = [instruction_id(self.robot, k) for k in range(len(PARAPHRASES[self.robot]))]
        if instructions == "train":
            self.instruction_choices = robot_ids[:N_TRAIN]
        elif instructions == "heldout":
            self.instruction_choices = robot_ids[N_TRAIN:]
        elif isinstance(instructions, int):
            self.instruction_choices = [robot_ids[instructions]]
        else:
            raise ValueError(f"instructions must be 'train', 'heldout' or an int, got {instructions!r}")
        self.sentences = all_instructions()
        self.seed = seed
        self._rng = None
        self._array = None  # memory-mapped vision.npy or images.npy, opened in each worker

    def __len__(self) -> int:
        return len(self.frames)

    def _frames_array(self):
        if self._array is None:
            name = "vision.npy" if self.mode == "cached" else "images.npy"
            self._array = np.load(self.robot_dir / name, mmap_mode="r")
        return self._array

    def reset_rng(self) -> None:
        """Restart the instruction choices from the seed (validation uses the same ones every time)."""
        self._rng = None

    def _random(self) -> np.random.Generator:
        if self._rng is None:  # one generator per DataLoader worker, seeded from torch (Phase 3 seeds it)
            info = torch.utils.data.get_worker_info()
            self._rng = np.random.default_rng(info.seed if info else self.seed)
        return self._rng

    def __getitem__(self, i: int) -> dict:
        t = int(self.frames[i])
        ep = int(self.episode_of_frame[t])
        start, end = int(self.episode_starts[ep]), int(self.episode_ends[ep])

        frame_ids = [max(t - 1, start), t]  # frame t repeated at the episode's first step
        k = min(CHUNK, end - t)              # real steps left in this episode
        actions = np.zeros((CHUNK, ACTION_DIM), dtype=np.float32)
        actions[:k] = self.action[t:t + k]

        lang_id = self.instruction_choices[self._random().integers(len(self.instruction_choices))]
        sample = {
            "lang": torch.from_numpy(self.language[lang_id]),
            "instruction": self.sentences[lang_id],
            "instruction_id": torch.tensor(lang_id, dtype=torch.int64),
            "state": torch.from_numpy(self.state[t].copy()),
            "embodiment_id": torch.tensor(EMBODIMENT_ID[self.robot], dtype=torch.int64),
            "actions": torch.from_numpy(actions),
            "action_dim_mask": self.action_dim_mask.clone(),
            "action_is_pad": torch.arange(CHUNK) >= k,
            "frame": torch.tensor(t, dtype=torch.int64),
            "episode": torch.tensor(ep, dtype=torch.int64),
        }
        frames = np.asarray(self._frames_array()[frame_ids])
        if self.mode == "cached":
            sample["vision"] = torch.from_numpy(frames.astype(np.float32))
        elif self.mode == "pixels":
            sample["pixels"] = torch.from_numpy(np.ascontiguousarray(frames))
        else:
            from vla.models.encoders import preprocess_image

            sample["images"] = preprocess_image(frames)
        return sample


class MixedBatchSampler(torch.utils.data.Sampler):
    """Endless batches of indices into ConcatDataset([robot 0 data, robot 1 data, ...]).

    Each batch takes a fixed share from every robot (spec: 50% PushT, 50% FetchPush), so neither
    robot dominates even though their datasets differ in size. Each robot's indices are shuffled
    and used once per pass, then reshuffled. The same seed gives the same batches.
    """

    def __init__(self, sizes, batch_size: int, fractions=None, seed: int = 0):
        self.sizes = [int(n) for n in sizes]
        if min(self.sizes) <= 0:
            raise ValueError(f"every robot needs at least one sample, got sizes {self.sizes}")
        fractions = fractions or [1 / len(self.sizes)] * len(self.sizes)
        counts = [int(round(batch_size * f)) for f in fractions[:-1]]
        self.counts = counts + [batch_size - sum(counts)]
        if min(self.counts) <= 0:
            raise ValueError(f"batch {batch_size} split as {self.counts}: every robot needs a share")
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)[:-1]]).tolist()
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        orders = [rng.permutation(n) for n in self.sizes]
        positions = [0] * len(self.sizes)
        while True:
            batch = []
            for r, (n, k) in enumerate(zip(self.sizes, self.counts)):
                for _ in range(k):
                    if positions[r] == n:
                        orders[r], positions[r] = rng.permutation(n), 0
                    batch.append(self.offsets[r] + int(orders[r][positions[r]]))
                    positions[r] += 1
            yield batch


def collate_mixed_pixels(samples: list[dict]) -> dict:
    """Collate for the VLA in pixels mode (Phase 6 adaptation runs).

    A VLA batch is half PushT (96x96 frames) and half FetchPush (128x128), and frames of different
    sizes cannot be stacked into one tensor. So every key except "pixels" is collated as usual, and
    "pixels" becomes a list of stacked tensors, one per run of consecutive same-size samples, in
    batch order (MixedBatchSampler puts all PushT first, then all FetchPush, so: 2 tensors). The model
    resizes each to 224 with preprocess_image() and joins them (AdaptedVLAPolicy.encode_pixels).
    """
    rest = torch.utils.data.default_collate([{k: v for k, v in s.items() if k != "pixels"} for s in samples])
    groups, run = [], []
    for s in samples:
        if run and s["pixels"].shape != run[-1].shape:
            groups.append(torch.stack(run))
            run = []
        run.append(s["pixels"])
    if run:
        groups.append(torch.stack(run))
    rest["pixels"] = groups
    return rest
