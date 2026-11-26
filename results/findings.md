# Findings (Phase 7)

Written on 2026-11-26 from `results/results.csv` and the tables in the README (both made by
`scripts/aggregate.py`). Every number below comes from those files, and the W&B run IDs are given
in brackets. **Status when written:** all 9 conditioning runs and all 12 baseline runs had finished.
Only 2 of the 9 adaptation runs had finished (seed 0 of lora16 and full_ft).

**How to read the gaps.** A gap between two models is called **real** only if its 95% paired-bootstrap
interval excludes 0. Otherwise it is **not distinguishable**. W&B does not hold per-seed results for
these runs (only totals such as 32/50), so every bootstrap below is the approximate one: it
resamples 5 blocks of 10 start seeds read from the console logs. Five units are too few for a
trustworthy interval. So an approximate interval that excludes 0 is reported as **provisional**, not
as real. The exact bootstrap over the 50 start seeds will replace it once the Kaggle `metrics.json`
files are downloaded (see "What would change these numbers").

**The FetchPush floor.** 5 of the 50 FetchPush starts (seeds 100004, 100010, 100021, 100036, 100049)
already have the cube within 5 cm of the target. A robot that does nothing therefore scores 10%. Every
FetchPush number below is compared with that 10%.

---

## 1. One model learned FetchPush well above the floor, but it barely solves PushT

- **FetchPush:** 59–68% success over 150 rollouts per variant (token 64.7%, film 59.3%, sep_heads
  68.0%). That is **+49 to +58 pp above the 10% floor**, and the pooled Wilson intervals (lowest bound
  51.3%) are far above 10%. It also holds when the 5 free starts are removed. Without per-seed data
  only a range is known, and even its worst case is 74/135 = 55% (film) on starts that need real
  pushing.
- **PushT:** 3–7% success (token 3.3%, film 7.3%, sep_heads 7.3%; pooled Wilson upper bounds 7.6–12.7%).
  The softer measure, mean max coverage, is about 0.47–0.52. So on average the policy gets the T about
  half-way onto the target, but rarely to the 95% needed for success.
- In short, the same network, data mix and training budget give a working FetchPush policy and a
  mostly failing PushT policy. PushT is the harder task here: it needs the T placed at the right angle
  as well as the right position, and it takes up to 300 steps instead of 50.

Runs: token `neo2m4ux` `vzqwyyqd` `yf3jvezd`, film `i0senai6` `2b1u0h6m` `houp0iwv`, sep_heads `m5gn67rl`
`lf76862i` `rgxuk3g9`.

## 2. The three ways of telling the model which robot it drives are not distinguishable

- Mean success: token 34.0 ± 2.6%, film 33.3 ± 4.7%, sep_heads 37.7 ± 8.3% (mean ± std over 3 seeds).
  The spread between seeds of one variant (up to ±8.3 pp) is as large as the gaps between variants.
- On mean success all three pairs are **not distinguishable**: token − film +0.7 pp [−8.0, +6.3],
  token − sep_heads −3.7 pp [−7.3, +0.0], film − sep_heads −4.3 pp [−9.0, +1.7].
- Two per-robot gaps have approximate intervals that exclude 0, so they are **provisional only**.
  sep_heads beats token on PushT by 4.0 pp [−7.3, −0.7 for token − sep_heads], and beats film on
  FetchPush by 8.7 pp [−16.0, −0.7]. These come from 5 blocks of starts, and the PushT one rests on
  11 vs 5 successes out of 150. Neither should be quoted as a result until the exact bootstrap confirms it.
- sep_heads was chosen for the adaptation runs by the pre-set rule (highest mean success). That choice
  was fine, but its lead is within noise: on this evidence, *which* conditioning method is used does
  not matter.

## 3. Baselines: BC-MLP collapsed to the average action; ResNet-18 is a strong specialist

- **BC-MLP on FetchPush collapsed to the mean action.** Its training loss drops from about 0.118 to
  0.098 during the 300 warmup steps, then stays flat for 3,800 steps: +0.1%, −0.6%, −0.6% change after
  warmup for seeds 0, 1, 2 (`lv456bp9` `qvkm8zhp` `4g9gno56`). ResNet-18 on the same data keeps falling,
  by about 40% (`4i1b97s2` `3oja5t5i` `nbzs4vx7`). This is why its val loss is 0.0974 in every seed: all
  three ended at the same "always output the average action" solution. BC-MLP's FetchPush success is
  exactly 5/50 in every seed = **the 10% floor, +0.0 pp**. The success video it saved is on seed
  100004, one of the already-solved starts. Its PushT success is 0/150. Details:
  [bc_mlp_fetch.md](bc_mlp_fetch.md), plot `plots/bc_mlp_fetch_loss.png`.
- The VLA (sep_heads) vs BC-MLP: +7.3 pp on PushT [+3.3, +10.7] and +58.0 pp on FetchPush
  [+46.7, +70.0]. The FetchPush gap is **real**. Its approximate interval is far from 0, and the
  pooled Wilson intervals do not even overlap (10.0% [6.2, 15.8] vs 68.0% [60.2, 74.9]), so this
  conclusion does not depend on how precise the bootstrap is.
- **ResNet-18 is a per-robot specialist, and it holds its own.** PushT: ResNet-18 14.7% vs the VLA's
  7.3%. The gap is −7.3 pp [−15.3, −0.7], which is **provisional** (approximate interval, and the Wilson
  intervals overlap: [9.9, 21.2] vs [4.1, 12.7]). FetchPush: VLA 68.0% vs ResNet-18 51.3% (+41 pp
  above the floor). The gap is +16.7 pp [−2.0, +37.3]: **not distinguishable**.
- So the shared VLA does not yet beat a separate ResNet-18 per robot. It may even be behind on PushT.

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

## 5. Adaptation (preliminary, 1 of 3 seeds): LoRA saves less memory than expected and scores lower

Only seed 0 of lora16 (`8z9ys2qa`) and full_ft (`rje18trq`) had finished. frozen_matched, the fair
frozen control, had not.

- Mean success: full fine-tuning 41.0% (PushT 22.0%, FetchPush 60.0%), LoRA 30.0% (PushT 14.0%,
  FetchPush 46.0%). **LoRA reached 73% of full fine-tuning** (PushT 64%, FetchPush 77%). The gap
  lora16 − full_ft is −11.0 pp [−15.0, −5.0] on the mean. That is **provisional**: one seed each, and
  the approximate bootstrap.
- Peak GPU memory at step 200: LoRA 6.73 GB vs full fine-tuning 8.71 GB = **23% lower**. That is a
  real but modest saving. The adapters themselves are tiny (442,368 vs 22,056,576 trainable DINOv2
  parameters), but both modes still run the full DINOv2 forward and backward pass over the images
  at batch 64. LoRA mainly saves the optimizer state and the weight gradients of DINOv2.
- Time: both took about 31 minutes per run, including evaluation (31.7 vs 31.1). LoRA saved no time.
- FetchPush vs the floor: full fine-tuning 60.0% (+50 pp), LoRA 46.0% (+36 pp).
- Something to watch, not a finding: full fine-tuning's 22% on PushT (11/50) is the best PushT rate of
  any VLA run so far. It is one seed, though. frozen_matched must also finish before we can say whether
  un-freezing helps, or whether the smaller batch at 2,500 steps does.

---

## What would change these numbers

- **Per-seed results.** Download the Kaggle output folders (each run's `metrics.json`) into
  `outputs/kaggle/` and rerun `.venv/bin/python scripts/aggregate.py`. The "approx." intervals then
  become exact paired bootstraps over the 50 start seeds. The FetchPush "unsolved starts" ranges then
  become exact counts, and the provisional gaps in findings 2, 3 and 5 are either confirmed or dropped.
  Runs from now on also save per-seed results to W&B (`eval/<robot>_episode_success`).
- **The adaptation batch.** Rerun the script when it finishes. Finding 5 should then be rewritten with
  3 seeds per mode and the frozen_matched control.
- The bootstrap treats each training seed as fixed: it measures luck in the start states, not luck in
  training. The mean ± std columns show training luck, and it is large (sep_heads PushT ranges 0–18%
  across seeds).
