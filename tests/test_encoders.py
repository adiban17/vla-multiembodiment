"""Phase 2 checks: preprocess_image(), patch pooling, DINOv2-small and MiniLM on CPU.

The model tests download facebook/dinov2-small (~88 MB) once into the Hugging Face cache.
"""

import numpy as np
import pytest
import torch

from vla.models.encoders import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    DinoV2Encoder,
    MiniLMEncoder,
    pool_patches,
    preprocess_image,
)


@pytest.mark.parametrize("size", [96, 128])
def test_preprocess_image_shapes_and_values(size):
    rng = np.random.default_rng(0)
    batch = rng.integers(0, 256, size=(3, size, size, 3), dtype=np.uint8)
    out = preprocess_image(batch)
    assert out.shape == (3, 3, 224, 224) and out.dtype == torch.float32
    assert preprocess_image(batch[0]).shape == (1, 3, 224, 224)                 # single HWC image
    torch.testing.assert_close(preprocess_image(torch.from_numpy(batch)), out)  # torch input
    torch.testing.assert_close(preprocess_image(torch.from_numpy(batch).permute(0, 3, 1, 2)), out)  # NCHW

    # A flat colour stays flat after resizing, and is ImageNet-normalized
    flat = np.full((size, size, 3), 128, dtype=np.uint8)
    expected = (128 / 255 - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
    torch.testing.assert_close(preprocess_image(flat)[0, :, 100, 100], expected)


def test_preprocess_image_rejects_float_input():
    with pytest.raises(TypeError):
        preprocess_image(np.zeros((96, 96, 3), dtype=np.float32))


def test_pool_patches_is_a_2x2_average():
    patches = torch.randn(2, 256, 384)
    pooled = pool_patches(patches)
    assert pooled.shape == (2, 64, 384)
    grid = patches.reshape(2, 16, 16, 384)
    manual = grid.reshape(2, 8, 2, 8, 2, 384).mean(dim=(2, 4)).reshape(2, 64, 384)
    torch.testing.assert_close(pooled, manual)


@pytest.fixture(scope="module")
def dino():
    return DinoV2Encoder()


def test_dinov2_runs_on_two_images_on_cpu(dino):
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(2, 96, 96, 3), dtype=np.uint8)
    with torch.inference_mode():
        out = dino(preprocess_image(images))
    assert out.shape == (2, 65, 384) and out.dtype == torch.float32 and torch.isfinite(out).all()
    assert not any(p.requires_grad for p in dino.parameters())  # frozen
    dino.train()
    assert not dino.model.training  # stays in eval mode while frozen


def test_minilm_gives_one_unit_vector_per_sentence():
    lm = MiniLMEncoder()
    with torch.inference_mode():
        out = lm(["push the T onto the target", "push the cube to the red target", "hello"])
    assert out.shape == (3, 384)
    torch.testing.assert_close(out.norm(dim=1), torch.ones(3))
