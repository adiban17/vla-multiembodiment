"""The feature cache: run the frozen encoders once over every frame and save the results.

Frozen-encoder training then reads 65 x 384 numbers per frame from disk instead of running
DINOv2 every step, which makes it many times faster. The cache folder is self-contained, so
later Kaggle notebooks attach only this one output:

    <run>/stats.json          per-robot normalization stats + the train/val episode split
    <run>/language.npy        (20, 384) float32 MiniLM vectors of all instructions
    <run>/instructions.json   the 20 sentences, in language.npy row order
    <run>/size_report.json    bytes and GB per file and robot
    <run>/<robot>/meta.json   kind "features", source dataset, image source, model revisions
    <run>/<robot>/lowdim.npz  state, action, episode_length, episode_seed (cached episodes only)
    <run>/<robot>/vision.npy  (N, 65, 384) float16, DINOv2 CLS + 8x8 pooled patches
    <run>/<robot>/images.npy  (N, H, W, 3) uint8, the raw frames, uncompressed

vision.npy and images.npy are plain .npy files opened memory-mapped: reading frame i jumps
straight to it on disk. (Compressed .npz must be unpacked whole, too slow for random access in
the adaptation runs, which read raw images.)

`images_for_training` is copied from the source dataset, so a cache built from Mac debug
images stays debug-only and WindowDataset refuses it for training.
"""

import json
from pathlib import Path

import numpy as np

from vla.data.dataset import split_episodes
from vla.data.instructions import N_TRAIN, PARAPHRASES, all_instructions
from vla.data.normalize import compute_robot_stats, save_stats
from vla.data.shards import read_meta, read_shards, write_meta

FEATURE_SHAPE = (65, 384)
SPEC_VISION_GB = 2.5  # docs/spec.md estimate for ~50k frames
ENCODE_CHUNK = 1024   # frames handed to encode_fn at a time


def _shard_paths(data_dir) -> list[Path]:
    return sorted(Path(data_dir).glob("shard_*.npz"))


def cache_robot(data_dir, out_dir, encode_fn, max_episodes: int | None = None, encoder_info: dict | None = None) -> dict:
    """Cache one robot's dataset. encode_fn: uint8 (n, H, W, 3) -> float (n, 65, 384)."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = read_meta(data_dir)
    if not src.get("has_images"):
        raise ValueError(f"{data_dir} has no images to cache")
    low = read_shards(data_dir, with_images=False)
    n_eps = len(low["episode_length"]) if max_episodes is None else min(max_episodes, len(low["episode_length"]))
    n_frames = int(low["episode_ends"][n_eps - 1])

    paths = _shard_paths(data_dir)
    with np.load(paths[0]) as first:
        image_shape = tuple(first["images"].shape[1:])
    images = np.lib.format.open_memmap(out_dir / "images.npy", mode="w+", dtype=np.uint8, shape=(n_frames, *image_shape))
    vision = np.lib.format.open_memmap(out_dir / "vision.npy", mode="w+", dtype=np.float16, shape=(n_frames, *FEATURE_SHAPE))

    offset = 0
    for path in paths:
        if offset >= n_frames:
            break
        with np.load(path) as shard:
            frames = shard["images"][: n_frames - offset]
        images[offset:offset + len(frames)] = frames
        for c in range(0, len(frames), ENCODE_CHUNK):
            part = frames[c:c + ENCODE_CHUNK]
            vision[offset + c:offset + c + len(part)] = np.asarray(encode_fn(part), dtype=np.float32).astype(np.float16)
        offset += len(frames)
        print(f"  {src['robot']}: {offset}/{n_frames} frames cached")
    images.flush()
    vision.flush()
    del images, vision

    np.savez(out_dir / "lowdim.npz",
             state=low["state"][:n_frames], action=low["action"][:n_frames],
             episode_length=low["episode_length"][:n_eps], episode_seed=low["episode_seed"][:n_eps])
    meta = {
        "kind": "features",
        "robot": src["robot"],
        "n_episodes": int(n_eps),
        "n_frames": n_frames,
        "max_episodes": max_episodes,
        "image_shape": list(image_shape),
        "feature_shape": list(FEATURE_SHAPE),
        "feature_dtype": "float16",
        "has_images": True,
        "image_source": src.get("image_source"),
        "images_for_training": bool(src.get("images_for_training")),
        "state_names": src.get("state_names"),
        "action_names": src.get("action_names"),
        "source_dir": str(data_dir),
        "source_meta": src,
        **(encoder_info or {}),
    }
    write_meta(out_dir, meta)
    return meta


def write_stats(run_dir, robots: list[str], val_frac: float = 0.1, split_seed: int = 0) -> dict:
    """Normalization stats from each robot's TRAIN episodes, plus the split itself, into stats.json."""
    run_dir = Path(run_dir)
    stats = {"split": {}}
    for robot in robots:
        low = np.load(run_dir / robot / "lowdim.npz")
        meta = read_meta(run_dir / robot)
        lengths = low["episode_length"]
        train, val = split_episodes(len(lengths), val_frac, split_seed)
        ends = np.cumsum(lengths)
        rows = np.concatenate([np.arange(ends[e] - lengths[e], ends[e]) for e in train])
        stats[robot] = compute_robot_stats(
            low["state"][rows], low["action"][rows],
            action_range=(-1.0, 1.0) if robot == "fetch" else None,  # FetchPush actions are left as they are
            state_names=meta.get("state_names"), action_names=meta.get("action_names"),
        )
        stats["split"][robot] = {"train": train, "val": val, "val_frac": val_frac, "seed": split_seed}
    save_stats(run_dir / "stats.json", stats)
    return stats


def print_stats(stats: dict) -> None:
    for robot, s in stats.items():
        if robot == "split":
            continue
        split = stats["split"][robot]
        print(f"{robot}: {len(split['train'])} train / {len(split['val'])} val episodes")
        for key in ("state", "action"):
            k = s[key]
            names = k.get("names") or [str(i) for i in range(len(k["min"]))]
            for name, lo, hi, const in zip(names, k["min"], k["max"], k["constant"]):
                note = "  CONSTANT -> 0" if const else ("  (fixed range)" if k["fixed_range"] else "")
                print(f"  {key:6s} {name:9s} min {lo:10.4f}  max {hi:10.4f}{note}")


def write_language(run_dir, language_fn) -> np.ndarray:
    """language_fn: list of sentences -> (n, 384). Saves language.npy and instructions.json."""
    sentences = all_instructions()
    vectors = np.asarray(language_fn(sentences), dtype=np.float32)
    if vectors.shape != (len(sentences), FEATURE_SHAPE[1]):
        raise ValueError(f"language vectors have shape {vectors.shape}")
    np.save(Path(run_dir) / "language.npy", vectors)
    (Path(run_dir) / "instructions.json").write_text(json.dumps(
        {"sentences": sentences, "paraphrases": PARAPHRASES, "n_train": N_TRAIN}, indent=2))
    return vectors


def size_report(run_dir, robots: list[str]) -> dict:
    run_dir = Path(run_dir)
    gb = lambda b: round(b / 1e9, 3)  # noqa: E731
    report = {}
    for robot in robots:
        d = run_dir / robot
        r = {"n_frames": read_meta(d)["n_frames"]}
        for name in ("vision", "images", "lowdim"):
            path = d / (f"{name}.npz" if name == "lowdim" else f"{name}.npy")
            r[f"{name}_bytes"] = path.stat().st_size
            r[f"{name}_gb"] = gb(path.stat().st_size)
        report[robot] = r
    for name in ("vision", "images"):
        total = sum(report[r][f"{name}_bytes"] for r in robots)
        report[f"total_{name}_bytes"] = total
        report[f"total_{name}_gb"] = gb(total)
    report["total_bytes"] = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file() and p.suffix in (".npy", ".npz", ".json"))
    report["total_gb"] = gb(report["total_bytes"])
    frames = sum(report[r]["n_frames"] for r in robots)
    report["total_frames"] = frames
    report["spec_vision_estimate_gb"] = SPEC_VISION_GB
    report["vision_gb_per_50k_frames"] = gb(report["total_vision_bytes"] / max(frames, 1) * 50_000)
    (run_dir / "size_report.json").write_text(json.dumps(report, indent=2))
    return report


def print_size_report(report: dict, robots: list[str]) -> None:
    for robot in robots:
        r = report[robot]
        print(f"  {robot:6s} {r['n_frames']:6d} frames | vision.npy {r['vision_gb']:.3f} GB | images.npy {r['images_gb']:.3f} GB")
    print(f"  total  {report['total_frames']:6d} frames | vision {report['total_vision_gb']:.3f} GB | images "
          f"{report['total_images_gb']:.3f} GB | all files {report['total_gb']:.3f} GB")
    print(f"  vision scaled to 50k frames: {report['vision_gb_per_50k_frames']:.2f} GB (spec estimate ~{SPEC_VISION_GB} GB)")


def build_cache(run_dir, data_dirs: dict, encode_fn, language_fn, max_episodes: int | None = None,
                encoder_info: dict | None = None, val_frac: float = 0.1, split_seed: int = 0) -> dict:
    """Cache every robot in data_dirs ({"pusht": dir, "fetch": dir}), then stats, language, size report."""
    run_dir = Path(run_dir)
    robots = list(data_dirs)
    metas = {robot: cache_robot(d, run_dir / robot, encode_fn, max_episodes, encoder_info) for robot, d in data_dirs.items()}
    stats = write_stats(run_dir, robots, val_frac, split_seed)
    write_language(run_dir, language_fn)
    report = size_report(run_dir, robots)
    return {"robots": metas, "stats": stats, "size_report": report}


def read_frames_from_shards(data_dir, indices) -> np.ndarray:
    """uint8 images of the given global frame indices, read from the compressed shards."""
    indices = np.asarray(indices)
    out, offset = {}, 0
    for path in _shard_paths(data_dir):
        with np.load(path) as shard:
            n = int(shard["episode_length"].sum())
            wanted = indices[(indices >= offset) & (indices < offset + n)]
            if len(wanted):
                images = shard["images"]
                for i in wanted:
                    out[int(i)] = images[i - offset]
        offset += n
    return np.stack([out[int(i)] for i in indices])


def verify_cache(robot_dir, data_dir, encode_fn, n_check: int = 32, seed: int = 0,
                 rtol: float = 2e-3, atol: float = 2e-3) -> dict:
    """Recompute features live for some frames (first, last and random ones) and compare.

    Also checks images.npy holds exactly the shard's pixels. fp16 keeps ~3 significant digits,
    so the cached and live features agree to about 0.05% of their size.
    """
    robot_dir = Path(robot_dir)
    vision = np.load(robot_dir / "vision.npy", mmap_mode="r")
    images = np.load(robot_dir / "images.npy", mmap_mode="r")
    n = len(vision)
    rng = np.random.default_rng(seed)
    idx = np.unique(np.concatenate([[0, n - 1], rng.choice(n, size=min(n_check, n), replace=False)]))
    frames = read_frames_from_shards(data_dir, idx)
    images_equal = bool(np.array_equal(np.asarray(images[idx]), frames))
    live = np.asarray(encode_fn(frames), dtype=np.float32)
    cached = np.asarray(vision[idx], dtype=np.float32)
    diff = np.abs(live - cached)
    within = diff <= atol + rtol * np.abs(live)
    return {
        "n_checked": int(len(idx)),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_feature": float(np.abs(live).max()),
        "fraction_within_tolerance": float(within.mean()),
        "images_equal": images_equal,
        "ok": bool(within.all() and images_equal),
    }
