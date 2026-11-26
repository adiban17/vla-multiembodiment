# Findings (Phase 7)

Written on 2026-11-27 from `results/results.csv` and the tables in the README (both made by
`scripts/aggregate.py`). Every number below comes from those files, and the W&B run IDs are given
in brackets. **Status:** all 30 planned runs have finished (9 conditioning, 12 baseline, 9 adaptation),
and all 30 have per-seed results from their Kaggle `metrics.json` (`episodes_source = metrics.json`).

**How to read the gaps.** A gap between two models is called **real** only if its exact 95% paired
bootstrap interval excludes 0. The bootstrap resamples the 50 start seeds that every run shares and
pools the 3 training seeds. If the interval includes 0 (or reaches exactly 0), the gap is **not
distinguishable**. Every interval in this file is exact. The approximate 5-block intervals and the
"provisional" labels of the earlier version are gone (see "What changed" at the end).

Two limits of the bootstrap. It treats the training seeds as fixed, so it measures luck in the start
states, not luck in training. Where the training seeds of two models overlap, that is said below.
It is also run on many pairs (26 gaps in the README). At 95%, about 1 in 20 true-zero gaps would
exclude 0 by chance, so a single borderline gap is weak evidence.

**The FetchPush floor.** 5 of the 50 FetchPush starts (seeds 100004, 100010, 100021, 100036, 100049)
already have the cube within 5 cm of the target. A robot that does nothing therefore scores 10%. Every
FetchPush number below is compared with that 10%, and the "unsolved starts" numbers count only the
other 45 starts per run (135 per configuration), where a do-nothing robot scores 0%.

---

## 1. One model learned FetchPush well above the floor, but it barely solves PushT

- **FetchPush:** 59–68% success over 150 rollouts per variant (token 64.7%, film 59.3%, sep_heads
  68.0%). That is **+49 to +58 pp above the 10% floor**, and the pooled Wilson intervals (lowest bound
  51.3%) are far above 10%. It also holds on the 45 unsolved starts alone: token 82/135 = 60.7%
  [52.3, 68.6], film 74/135 = 54.8% [46.4, 63.0], sep_heads 88/135 = 65.2% [56.8, 72.7]
  (95% Wilson). These are starts where the robot really has to push the cube.
- **PushT:** 3–7% success (token 3.3%, film 7.3%, sep_heads 7.3%; pooled Wilson upper bounds 7.6–12.7%).
  The softer measure, mean max coverage, is about 0.47–0.52. So on average the policy gets the T about
  half-way onto the target, but rarely to the 95% needed for success.
- In short, the same network, data mix and training budget give a working FetchPush policy and a
  mostly failing PushT policy. PushT is the harder task here: it needs the T placed at the right angle
  as well as the right position, and it takes up to 300 steps instead of 50.

Runs: token `neo2m4ux` `vzqwyyqd` `yf3jvezd`, film `i0senai6` `2b1u0h6m` `houp0iwv`, sep_heads `m5gn67rl`
`lf76862i` `rgxuk3g9`.

## 2. The three ways of telling the model which robot it drives: one real gap, and it is small

- Mean success: token 34.0 ± 2.6%, film 33.3 ± 4.7%, sep_heads 37.7 ± 8.3% (mean ± std over 3 seeds).
  The spread between seeds of one variant (up to ±8.3 pp) is as large as the gaps between variants.
- Exact gaps (A − B, pp [95% interval]):

  | Pair | PushT | FetchPush | Mean | Verdict |
  |---|---|---|---|---|
  | token − film | −4.0 [−9.3, +1.3] | +5.3 [−3.3, +14.0] | +0.7 [−4.3, +5.3] | not distinguishable |
  | token − sep_heads | −4.0 [−8.7, +0.0] | −3.3 [−10.7, +4.0] | −3.7 [−7.3, +0.0] | not distinguishable |
  | film − sep_heads | +0.0 [−6.0, +6.0] | **−8.7 [−16.7, −0.7]** | −4.3 [−9.0, +0.3] | real on FetchPush only |

- **sep_heads beats film on FetchPush by 8.7 pp. This is the only real gap.** It holds on the
  unsolved starts too (65.2% vs 54.8%). But it is weak: the upper end is −0.7 pp, it is 1 of 9
  conditioning gaps, and the training seeds overlap (sep_heads 62, 66, 76%; film 64, 56, 58%: film's
  seed 0 beats sep_heads' seed 0).
- **sep_heads vs token is not distinguishable.** The PushT and mean intervals reach exactly 0
  (upper end +0.0), so they do not exclude it. The earlier provisional claim "sep_heads beats token on
  PushT" is dropped.
- No pair differs on mean success. sep_heads was chosen for the adaptation runs by the pre-set rule
  (highest mean success). That choice was fine, but its lead is within noise: on this evidence,
  *which* conditioning method is used barely matters.

## 3. Baselines: BC-MLP collapsed to the average action; the VLA beats ResNet-18 on FetchPush but loses on PushT

- **BC-MLP on FetchPush collapsed to the mean action.** Its training loss drops from about 0.118 to
  0.098 during the 300 warmup steps, then stays flat for 3,800 steps: +0.1%, −0.6%, −0.6% change after
  warmup for seeds 0, 1, 2 (`lv456bp9` `qvkm8zhp` `4g9gno56`). ResNet-18 on the same data keeps falling,
  by about 40% (`4i1b97s2` `3oja5t5i` `nbzs4vx7`). This is why its val loss is 0.0974 in every seed: all
  three ended at the same "always output the average action" solution. BC-MLP's FetchPush success is
  exactly 5/50 in every seed = **the 10% floor, +0.0 pp**, and the per-seed results now confirm that
  its 5 wins are exactly the 5 already-solved starts in all three seeds: **0/135 on unsolved starts**.
  Its PushT success is 0/150. Details: [bc_mlp_fetch.md](bc_mlp_fetch.md), plot `plots/bc_mlp_fetch_loss.png`.
- **The VLA (sep_heads) vs BC-MLP: both gaps are real.** +7.3 pp on PushT [+4.0, +11.3] and +58.0 pp on
  FetchPush [+47.3, +68.7].
- **The VLA (sep_heads, frozen) vs ResNet-18 on FetchPush: +16.7 pp [+3.3, +30.0]. This gap is real.**
  The VLA scores 68.0% (102/150) and the ResNet-18 specialist 51.3% (77/150). On the 45 unsolved starts:
  88/135 = 65.2% [56.8, 72.7] vs 68/135 = 50.4% [42.0, 58.7]. Part of the gap comes from the
  already-solved starts: the VLA keeps 14 of its 15 free wins, while ResNet-18 keeps only 9 of 15 (it
  sometimes pushes a cube that is already on target away). On the unsolved starts alone the same
  paired bootstrap gives +14.8 pp [+0.7, +28.9] (computed from the per-seed strings in results.csv), so
  the gap survives but with less room. Every VLA seed (62, 66, 76%) beats every ResNet-18 seed
  (52, 52, 50%), so this is not one lucky seed. The earlier approximate version called this gap
  "not distinguishable"; the exact bootstrap shows it is real.
- **The VLA vs ResNet-18 on PushT: −7.3 pp [−14.0, −0.7]. Also real, in ResNet-18's favor.**
  ResNet-18 14.7% (22/150, Wilson [9.9, 21.2]) vs the VLA 7.3% (11/150, [4.1, 12.7]). Its upper end
  is close to 0 and the seeds overlap (VLA 0, 4, 18%; ResNet-18 16, 20, 8%: the VLA's seed 2 beats two
  ResNet-18 seeds).
- So with frozen features, the one shared VLA is **better than a per-robot ResNet-18 on FetchPush and
  worse on PushT**. The comparison is not budget-matched: both train 4,000 steps, but the VLA uses
  batch 256 shared by two robots and ResNet-18 batch 128 on one robot.

## 4. Failure modes seen in the rollout videos

I watched the saved videos of the sep_heads runs (each run saves a success, a failure and one
fixed "random" start per robot). Four ways the policy fails showed up:

1. **PushT: right place, wrong angle, then it stops trying.** In `lf76862i` (seed 1), start seed
   100042, the agent pushes the T onto the target within the first half of the episode, but rotated
   about 90° the wrong way. For the second half (about 130 steps) it sits just below the T and barely moves.
   In `rgxuk3g9` (seed 2), start 100000, the T starts almost on the target. The agent hovers around it
   for most of the episode, then a late push turns it further off. Position is learned better than
   orientation, and the policy has no clear "fix the angle" behavior.
2. **PushT: pushing the T into the wall and getting stuck.** In `m5gn67rl` (seed 0), start 100001, the
   agent pushes the T away from the target into the bottom-left corner in the first few seconds, then
   stays pinned against the wall for the rest of the episode.
3. **FetchPush: the cube slides off the side of the gripper.** In `rgxuk3g9`, start 100000, the arm
   reaches the cube and starts pushing. Then the cube slips off to one side, and the arm carries on
   to the target without it. The cube ends a few centimetres short and to the side, outside the 5 cm
   success radius.
4. **FetchPush: overshoot, sometimes with the arm hiding the scene.** In `m5gn67rl`, start 100000, the
   cube is pushed past the target. In `lf76862i`, start 100042, the arm comes down in front of the
   45° camera and covers both the cube and the red target for most of the episode. The policy then
   has to act on what it remembers, not what it sees (the camera note in CLAUDE.md predicted this).

A likely cause of (1) and (2), not tested here: the model is trained with an L1 loss to predict *one*
action chunk. Where the demonstrations disagree (go left or right around the T), it learns an average
of them, which looks like hesitation or a push in a poor direction.

## 5. Adaptation: un-freezing DINOv2 is what helps PushT; LoRA is clearly behind full fine-tuning only on PushT

All 9 runs finished: sep_heads, 2,500 steps, batch 64, seeds 0–2. frozen_matched is the fair control: same
batch and steps, encoder frozen. Runs: frozen_matched `516mcmj3` `981owyth` `cw33zzpp`, lora16 `8z9ys2qa`
`it8rjzpp` `qti0vdx9`, full_ft `rje18trq` `68prcuw5` `xl5yjno6`. ResNet-18: PushT `331zxjun` `8un5xk7w`
`ijmsr80q`, FetchPush `4i1b97s2` `3oja5t5i` `nbzs4vx7`.

| Mode | PushT | FetchPush (floor 10%) | FetchPush, unsolved starts | Mean | Peak GPU memory @200 | Minutes/run |
|---|---|---|---|---|---|---|
| frozen_matched | 1.3 ± 1.2% (2/150) | 46.7 ± 3.1% (+36.7 pp) | 56/135 = 41.5% [33.5, 49.9] | 24.0 ± 1.7% | 1.47 GB | 9.6 |
| lora16 | 9.3 ± 6.4% (14/150) | 59.3 ± 13.0% (+49.3 pp) | 75/135 = 55.6% [47.1, 63.7] | 34.3 ± 6.7% | 6.73 GB | 30.1 |
| full_ft | 22.7 ± 3.1% (34/150) | 53.3 ± 8.3% (+43.3 pp) | 66/135 = 48.9% [40.6, 57.2] | 38.0 ± 5.2% | 8.71 GB | 29.7 |

Exact gaps (A − B, pp [95% interval]):

| Pair | PushT | FetchPush | Mean | Verdict |
|---|---|---|---|---|
| lora16 − full_ft | **−13.3 [−20.7, −6.7]** | +6.0 [−4.7, +16.7] | −3.7 [−10.7, +3.0] | real on PushT only |
| lora16 − frozen_matched | **+8.0 [+4.0, +12.7]** | **+12.7 [+1.3, +23.3]** | **+10.3 [+4.7, +16.0]** | real on all three |
| full_ft − frozen_matched | **+21.3 [+13.3, +29.3]** | +6.7 [−5.3, +19.3] | **+14.0 [+7.7, +20.7]** | real on PushT and mean |
| full_ft − resnet18 | +8.0 [−1.3, +17.3] | +2.0 [−10.7, +15.3] | – | not distinguishable |
| lora16 − resnet18 | −5.3 [−12.0, +1.3] | +8.0 [−4.7, +20.7] | – | not distinguishable |

- **lora16 vs full_ft.** LoRA reaches 90% of full fine-tuning's mean success (34.3% vs 38.0%), 41% on
  PushT (9.3% vs 22.7%) and 111% on FetchPush (59.3% vs 53.3%). Only the PushT gap is real:
  −13.3 pp [−20.7, −6.7], and every full_ft seed (22, 20, 26%) beats every lora16 seed (14, 12, 2%).
  The mean gap (−3.7 pp) and the FetchPush gap (+6.0 pp) are **not distinguishable**, so "90%" and
  "111%" are within noise. The earlier provisional "LoRA is ahead on FetchPush" and "behind on the
  mean" are both dropped. The honest summary: **LoRA loses clearly on PushT and ties on FetchPush.**
- **full_ft vs the frozen control.** PushT +21.3 pp and mean +14.0 pp are real; every full_ft seed beats
  every frozen_matched seed (0, 2, 2%) on PushT. Letting DINOv2 adapt is what makes PushT work at all at
  this budget. FetchPush +6.7 pp is not distinguishable.
- **lora16 vs the frozen control: real on PushT, FetchPush and the mean** (earlier provisional, now
  confirmed). LoRA helps over frozen features. The training seeds still overlap a little: on PushT
  lora16's seed 2 (2%) ties the control's best seeds (2%), and on FetchPush lora16's seed 0 (46%) is
  inside the control's range (44–50%).
- **Memory: LoRA uses 23% less peak GPU memory than full fine-tuning (6.73 vs 8.71 GB at step 200).**
  It uses 4.6 times more than frozen features (1.47 GB). The adapters are tiny (442,368 vs 22,056,576
  trainable DINOv2 parameters). But both modes still run DINOv2 forward and backward on every image,
  so the activation memory stays. LoRA only saves DINOv2's weight gradients and optimizer state. Time:
  LoRA and full fine-tuning both take about 30 minutes per run (frozen: about 10), so LoRA saves no time.
- **Adapted VLA vs the ResNet-18 specialist: not distinguishable on either robot**, for both full_ft
  and lora16. So the fine-tuned VLA closes the PushT gap that the frozen VLA had (finding 3), but does
  not beat ResNet-18; and at this smaller budget it also loses the FetchPush lead that the frozen VLA
  had at 4,000 steps × 256. Not budget-matched: ResNet-18 trained 4,000 steps at batch 128 on one robot,
  the adapted VLA 2,500 steps at batch 64 shared by both robots.
- **The training budget matters as much as the encoder.** frozen_matched (2,500 steps × 64 = 160k
  samples) scores 24.0% mean, against 37.7% for the same frozen sep_heads model at 4,000 steps × 256
  (1.0M samples). On PushT, full_ft at the small budget (22.7%) beats frozen sep_heads at the large
  one (7.3%). That comparison mixes budget and encoder, so it is an observation, not a measured gap.

---

## What changed from the 2026-11-26 version (approximate → exact bootstrap)

- Confirmed real: sep_heads > film on FetchPush; lora16 > frozen_matched on PushT, FetchPush and mean;
  VLA sep_heads < ResNet-18 on PushT.
- Newly real: VLA sep_heads > ResNet-18 on FetchPush (was "not distinguishable").
- Dropped (now not distinguishable): sep_heads > token on PushT; lora16 > full_ft on FetchPush;
  lora16 < full_ft on the mean.
- Still real: full_ft > frozen_matched and full_ft > lora16 on PushT; VLA > BC-MLP on both robots.
- The FetchPush "unsolved starts" ranges are now exact counts, and BC-MLP's 5 wins per seed are
  confirmed to be exactly the 5 already-solved starts.

## What would change these numbers

- **More training seeds.** The bootstrap only covers start-state luck. Several real gaps (sep_heads −
  film on FetchPush, VLA − ResNet-18 on PushT, lora16 − frozen_matched) have overlapping training seeds,
  and with 3 seeds per model the seed-to-seed spread is large (sep_heads PushT ranges 0–18%).
- **More start seeds.** 50 starts per robot give intervals about ±7–15 pp wide; borderline gaps such as
  sep_heads − film (upper end −0.7 pp) or token − sep_heads (upper end +0.0 pp) could flip either way.
- **Budget-matched comparisons.** The adapted runs (2,500 × 64) and the baselines (4,000 × 128) and
  frozen VLA (4,000 × 256) use different budgets, so VLA vs ResNet-18 and frozen vs adapted across
  batches are not like-for-like.
