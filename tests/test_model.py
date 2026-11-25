"""Phase 3 checks: the trunk, the 4 conditioning variants, the policy and its loss.

Written before the Phase 3 code. Full-size models are only run forward (fast on CPU); the
"overfit one batch" checks use the tiny config so they finish in seconds on the Mac.
"""

import pytest
import torch

from vla.models.conditioning import VARIANTS
from vla.models.policy import balanced_l1, build_policy, count_params, plain_masked_l1
from vla.models.trunk import Trunk
from vla.train import make_optimizer, make_scheduler
from vla.utils.config import load_config

B = 8


def fake_batch(b: int = B, seed: int = 0) -> dict:
    """Half PushT (id 0), half FetchPush (id 1), shaped like WindowDataset batches.

    PushT has 2 real action dims (dims 2,3 are padding, set to 0) and 2 real state dims.
    Two samples have padded steps at the end of the chunk (past the episode end, actions 0).
    """
    g = torch.Generator().manual_seed(seed)
    emb = torch.tensor([0] * (b // 2) + [1] * (b - b // 2))
    dim_mask = torch.ones(b, 4, dtype=torch.bool)
    dim_mask[emb == 0, 2:] = False
    is_pad = torch.zeros(b, 16, dtype=torch.bool)
    is_pad[b // 2 - 1, 11:] = True
    is_pad[b - 1, 6:] = True
    actions = torch.rand(b, 16, 4, generator=g) * 2 - 1
    actions = actions * dim_mask[:, None, :] * ~is_pad[:, :, None]
    state = torch.rand(b, 8, generator=g) * 2 - 1
    state[emb == 0, 2:] = 0
    lang = torch.nn.functional.normalize(torch.randn(b, 384, generator=g), dim=-1)
    return {
        "vision": torch.randn(b, 2, 65, 384, generator=g),
        "lang": lang,
        "state": state,
        "embodiment_id": emb,
        "actions": actions,
        "action_dim_mask": dim_mask,
        "action_is_pad": is_pad,
    }


def full_policy(cond: str):
    torch.manual_seed(0)
    return build_policy(load_config([f"cond={cond}"]))


def tiny_policy(cond: str):
    torch.manual_seed(0)
    cfg = load_config([f"cond={cond}", "size=tiny"])
    return build_policy(cfg), cfg


# --- trunk ------------------------------------------------------------------


def test_trunk_shape_and_zero_film_is_a_no_op():
    torch.manual_seed(0)
    trunk = Trunk(depth=2, width=64, heads=4, ffn=128, dropout=0.0).eval()
    x = torch.randn(3, 20, 64)
    out = trunk(x)
    assert out.shape == (3, 20, 64)
    zero_film = torch.zeros(3, 2, 4, 64)  # (B, depth, [gamma1, beta1, gamma2, beta2], width)
    assert torch.equal(trunk(x, film=zero_film), out)
    assert not torch.allclose(trunk(x, film=torch.randn(3, 2, 4, 64)), out)


# --- policy shapes ------------------------------------------------------------


@pytest.mark.parametrize("cond", VARIANTS)
def test_full_size_forward_shapes(cond):
    policy = full_policy(cond).eval()
    batch = fake_batch(2)
    tokens = policy.embed(batch)
    assert tokens.shape == (2, 149 if cond == "token" else 148, 384)
    with torch.no_grad():
        pred = policy(batch)
    assert pred.shape == (2, 16, 4)
    assert torch.isfinite(pred).all()


def test_sep_heads_pads_pusht_outputs_with_zeros():
    policy = full_policy("sep_heads").eval()
    batch = fake_batch(4)
    with torch.no_grad():
        pred = policy(batch)
    pusht = batch["embodiment_id"] == 0
    assert torch.equal(pred[pusht][..., 2:], torch.zeros_like(pred[pusht][..., 2:]))
    assert (pred[~pusht][..., 2:] != 0).any()


def _outputs_for_both_ids(policy):
    batch = fake_batch(2)
    batch = {k: v[:1].repeat(2, *[1] * (v.ndim - 1)) for k, v in batch.items()}  # same sample twice
    batch["embodiment_id"] = torch.tensor([0, 1])
    with torch.no_grad():
        pred = policy.eval()(batch)
    return pred[0], pred[1]


@pytest.mark.parametrize("cond", VARIANTS)
def test_robot_identity_is_used_only_by_conditioned_variants(cond):
    policy = full_policy(cond)
    a, b = _outputs_for_both_ids(policy)
    if cond == "none":
        assert torch.equal(a, b)
    elif cond == "film":
        assert torch.equal(a, b), "FiLM starts at zero, so at init both robots get the same output"
        with torch.no_grad():
            policy.film.proj.weight.normal_(0, 0.02)
        a, b = _outputs_for_both_ids(policy)
        assert not torch.allclose(a, b)
    else:
        assert not torch.allclose(a, b)


# --- loss ---------------------------------------------------------------------


def _two_sample_case(err_pusht: float, err_fetch: float):
    """One PushT sample (2 real dims) and one FetchPush sample (4 real dims), no padded steps.
    Every valid element of a robot has the same absolute error."""
    target = torch.zeros(2, 16, 4)
    pred = torch.zeros(2, 16, 4)
    pred[0, :, :2] = err_pusht
    pred[0, :, 2:] = 123.0  # padded dims: must be ignored
    pred[1] = err_fetch
    dim_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    is_pad = torch.zeros(2, 16, dtype=torch.bool)
    return pred, target, dim_mask, is_pad, torch.tensor([0, 1])


def test_plain_masked_l1_matches_the_spec_formula():
    pred, target, dim_mask, is_pad, _ = _two_sample_case(1.0, 0.0)
    # 32 PushT elements with error 1, 64 FetchPush elements with error 0
    assert plain_masked_l1(pred, target, dim_mask, is_pad).item() == pytest.approx(32 / 96)


def test_balanced_loss_gives_each_robot_equal_weight():
    """The plain mean counts elements, so FetchPush (4 dims) outweighs PushT (2 dims) 2 to 1.
    The balanced loss averages the two per-robot means instead."""
    # Same error everywhere: both methods agree
    args = _two_sample_case(0.3, 0.3)
    assert plain_masked_l1(*args[:4]).item() == pytest.approx(0.3)
    assert balanced_l1(*args)[0].item() == pytest.approx(0.3)

    # Only PushT is wrong: plain says 1/3, balanced says 1/2
    args = _two_sample_case(1.0, 0.0)
    assert plain_masked_l1(*args[:4]).item() == pytest.approx(1 / 3)
    loss, sums, counts = balanced_l1(*args)
    assert loss.item() == pytest.approx(0.5)
    assert counts.tolist() == [32, 64]
    assert (sums / counts).tolist() == pytest.approx([1.0, 0.0])

    # Only FetchPush is wrong: plain says 2/3, balanced still says 1/2 (symmetric)
    args = _two_sample_case(0.0, 1.0)
    assert plain_masked_l1(*args[:4]).item() == pytest.approx(2 / 3)
    assert balanced_l1(*args)[0].item() == pytest.approx(0.5)


def test_balanced_loss_gradient_is_split_half_and_half():
    """Total gradient weight per robot is 0.5, even with 3 PushT samples and 1 FetchPush sample."""
    torch.manual_seed(0)
    emb = torch.tensor([0, 0, 0, 1])
    dim_mask = torch.ones(4, 4, dtype=torch.bool)
    dim_mask[:3, 2:] = False
    is_pad = torch.zeros(4, 16, dtype=torch.bool)
    is_pad[1, 10:] = True
    pred = (torch.rand(4, 16, 4) + 0.5).requires_grad_()  # error > 0 everywhere, so |grad| = weight
    loss, _, _ = balanced_l1(pred, torch.zeros(4, 16, 4), dim_mask, is_pad, emb)
    loss.backward()
    weight = pred.grad.abs()
    assert weight[:3].sum().item() == pytest.approx(0.5)
    assert weight[3].sum().item() == pytest.approx(0.5)
    assert weight[:3, :, 2:].sum().item() == 0  # padded dims
    assert weight[1, 10:].sum().item() == 0     # padded steps


def test_balanced_loss_with_one_robot_is_that_robots_mean():
    pred, target, dim_mask, is_pad, emb = _two_sample_case(0.25, 0.0)
    loss, _, _ = balanced_l1(pred[:1], target[:1], dim_mask[:1], is_pad[:1], emb[:1])
    assert loss.item() == pytest.approx(0.25)


def test_policy_loss_ignores_padding():
    policy, _ = tiny_policy("film")
    policy.eval()
    batch = fake_batch()
    base = policy.loss(batch)["loss"].item()
    changed = dict(batch)
    changed["actions"] = batch["actions"].clone()
    pad = ~batch["action_dim_mask"][:, None, :] | batch["action_is_pad"][:, :, None]
    changed["actions"][pad.expand_as(changed["actions"])] = 99.0
    assert policy.loss(changed)["loss"].item() == pytest.approx(base)


# --- parameter counts ------------------------------------------------------------


def test_trainable_parameter_counts_match_the_spec(capsys):
    counts = {cond: count_params(full_policy(cond)) for cond in VARIANTS}
    with capsys.disabled():
        print("\nTrainable parameters (full size):")
        for cond, n in counts.items():
            print(f"  {cond:10s} {n:>12,d}  ({n / 1e6:.2f}M, {n - counts['none']:+,d} vs none)")
    assert 14.4e6 < counts["token"] < 15.0e6          # spec: about 14.7M
    assert counts["token"] - counts["none"] == 768    # 2 robots x 384
    assert 0.75e6 < counts["film"] - counts["token"] < 0.85e6   # spec: about 0.8M (FiLM vs none, +768 for token)
    assert 0.14e6 < counts["sep_heads"] - counts["none"] < 0.16e6  # spec: about 0.15M


# --- overfit one batch ----------------------------------------------------------


@pytest.mark.parametrize("cond", VARIANTS)
def test_each_variant_overfits_one_batch(cond):
    policy, cfg = tiny_policy(cond)
    steps = 300
    optimizer = make_optimizer(policy, cfg.optim)
    scheduler = make_scheduler(optimizer, cfg.optim.warmup_steps, steps)
    batch = fake_batch()
    policy.train()
    losses = []
    for _ in range(steps):
        loss = policy.loss(batch)["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.optim.grad_clip)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    print(f"{cond}: loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < 0.02
    assert losses[-1] < 0.05 * losses[0]
