"""Phase 2 checks: the real DINOv2 feature cache on tiny data, on CPU.

Cached features (fp16, memory-mapped) must equal features computed live for the same frame,
within fp16 tolerance. This is the guard against a train/eval preprocessing mismatch.
"""

import json

import numpy as np
import pytest

from conftest import _write_robot, fake_language
from vla.data.cache_features import FEATURE_SHAPE, build_cache, verify_cache
from vla.data.dataset import WindowDataset
from vla.data.shards import read_shards
from vla.models.encoders import DinoV2Encoder, encode_frames


@pytest.fixture(scope="module")
def encode():
    dino = DinoV2Encoder()
    return lambda frames: encode_frames(dino, frames, device="cpu", batch_size=4)


@pytest.fixture
def tiny_data(tmp_path):
    """2 short episodes per robot with random images (DINOv2 on ~13 frames is quick on CPU)."""
    dirs = {"pusht": _write_robot(tmp_path / "p", "pusht", [4, 3], 96, True),
            "fetch": _write_robot(tmp_path / "f", "fetch", [3, 3], 128, True)}
    rng = np.random.default_rng(0)
    for robot, d in dirs.items():  # replace the flat synthetic images with random ones, so features differ
        for path in sorted(d.glob("shard_*.npz")):
            arrays = dict(np.load(path))
            arrays["images"] = rng.integers(0, 256, size=arrays["images"].shape, dtype=np.uint8)
            np.savez_compressed(path, **arrays)
    return dirs


def test_cached_features_equal_live_features(tmp_path, tiny_data, encode):
    run = tmp_path / "cache"
    summary = build_cache(run, tiny_data, encode, fake_language)
    for robot, data_dir in tiny_data.items():
        vision = np.load(run / robot / "vision.npy", mmap_mode="r")
        assert isinstance(vision, np.memmap)
        assert vision.dtype == np.float16 and vision.shape == (len(read_shards(data_dir, False)["state"]), *FEATURE_SHAPE)

        # Live: the same frames straight from the shards, through the same preprocess + encoder
        frames = read_shards(data_dir)["images"]
        live = encode(frames)
        np.testing.assert_allclose(vision.astype(np.float32), live, rtol=2e-3, atol=2e-3)

        report = verify_cache(run / robot, data_dir, encode, n_check=5)
        assert report["ok"], report
        assert report["images_equal"]

        ds = WindowDataset(run / robot, split="all", mode="cached")
        assert ds[0]["vision"].shape == (2, 65, 384)

    sizes = json.loads((run / "size_report.json").read_text())
    for robot in tiny_data:
        assert sizes[robot]["vision_bytes"] > 0 and sizes[robot]["images_bytes"] > 0
    assert summary["size_report"]["total_bytes"] == sizes["total_bytes"]
