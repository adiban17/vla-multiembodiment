# Findings (Phase 7)

Written on 2026-11-26 from `results/results.csv` and the tables in the README (both made by
`scripts/aggregate.py`). Every number below comes from those files, and the W&B run IDs are given
in brackets. **Status:** all 30 planned runs have finished (9 conditioning, 12 baseline, 9 adaptation).

**How to read the gaps.** A gap between two models is called **real** only if its 95% paired-bootstrap
interval excludes 0. Otherwise it is **not distinguishable**. Per-seed results are still not available:
W&B holds only totals such as 32/50 for these runs, and the downloaded Kaggle notebooks
(`outputs/kaggle/*.ipynb`) hold only the console output, not the run folders' `metrics.json`. So every
bootstrap below is the approximate one: it resamples 5 blocks of 10 start seeds read from the console
logs. Five units are too few for a trustworthy interval, so an approximate interval that excludes 0 is
**provisional**, not real. There is one exception. A gap is called real when the approximate interval
excludes 0, the pooled Wilson intervals do not overlap, *and* every training seed of one model beats
every seed of the other. Such a gap is too large for the bootstrap's precision to matter. The exact
bootstrap over the 50 start seeds will replace all of this once the `metrics.json` files are downloaded
(see "What would change these numbers").

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
- So the frozen VLA does not beat a separate ResNet-18 per robot, and it may be behind on PushT.
  Fine-tuning DINOv2 changes the PushT picture (finding 5): full_ft reaches 22.7% PushT, and its gap to
  ResNet-18 becomes not distinguishable.

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

## 5. Adaptation: un-freezing DINOv2 is what helps PushT; LoRA keeps 90% of it overall, but only 41% on PushT

All 9 runs finished: sep_heads, 2,500 steps, batch 64, seeds 0–2. frozen_matched is the fair control: same
batch and steps, encoder frozen. Runs: frozen_matched `516mcmj3` `981owyth` `cw33zzpp`, lora16 `8z9ys2qa`
`it8rjzpp` `qti0vdx9`, full_ft `rje18trq` `68prcuw5` `xl5yjno6`. ResNet-18 PushT: `331zxjun` `8un5xk7w` `ijmsr80q`.

| Mode | PushT | FetchPush (floor 10%) | Mean | Peak GPU memory @200 | Minutes/run |
|---|---|---|---|---|---|
| frozen_matched | 1.3 ± 1.2% (2/150) | 46.7 ± 3.1% (+36.7 pp) | 24.0 ± 1.7% | 1.47 GB | 9.6 |
| lora16 | 9.3 ± 6.4% (14/150) | 59.3 ± 13.0% (+49.3 pp) | 34.3 ± 6.7% | 6.73 GB | 30.1 |
| full_ft | 22.7 ± 3.1% (34/150) | 53.3 ± 8.3% (+43.3 pp) | 38.0 ± 5.2% | 8.71 GB | 29.7 |

- **PushT: full fine-tuning is clearly better than the frozen control. This gap is real.** It is
  +21.3 pp [+18.0, +26.0]. The pooled Wilson intervals do not overlap ([16.7, 30.0] vs [0.4, 4.7]), and
  every full_ft seed (22, 20, 26%) beats every frozen_matched seed (0, 2, 2%). Letting DINOv2 adapt is
  what makes PushT work at all at this budget.
- **PushT: LoRA reaches only 41% of full fine-tuning (9.3% vs 22.7%). This gap is also real.** It is
  −13.3 pp [−16.0, −10.7]. The Wilson intervals do not overlap ([5.6, 15.1] vs [16.7, 30.0]), and every
  full_ft seed beats every lora16 seed (14, 12, 2%). LoRA vs the frozen control on PushT is +8.0 pp
  [+5.3, +11.3]. That one is **provisional**: lora16's seed 2 (2%) ties the control.
- **FetchPush: no mode is clearly ahead.** LoRA 59.3%, full_ft 53.3%, frozen_matched 46.7%, all well
  above the 10% floor. **LoRA reaches 111% of full fine-tuning here.** The gaps lora16 − full_ft +6.0 pp
  [+2.0, +10.0] and lora16 − frozen_matched +12.7 pp [+1.3, +20.7] are **provisional**: their seeds
  overlap a lot (lora16 46, 72, 60%; full_ft 60, 44, 56%). full_ft − frozen_matched +6.7 pp [−5.3, +16.7]
  is **not distinguishable**. Unsolved-start ranges: 55–70 / 74–89 / 65–80 of 135.
- **Mean success: LoRA reaches 90% of full fine-tuning (34.3% vs 38.0%).** That 90% is an average of a
  big PushT loss and a small FetchPush gain, so per robot (41% and 111%) is the honest way to report it.
  The mean gap −3.7 pp [−6.3, −1.0] is **provisional**.
- **Memory: LoRA uses 23% less peak GPU memory than full fine-tuning (6.73 vs 8.71 GB at step 200).**
  It uses 4.6 times more than frozen features (1.47 GB). The adapters are tiny (442,368 vs 22,056,576
  trainable DINOv2 parameters). But both modes still run DINOv2 forward and backward on every image,
  so the activation memory stays. LoRA only saves DINOv2's weight gradients and optimizer state. Time:
  LoRA and full fine-tuning both take about 30 minutes per run (frozen: about 10), so LoRA saves no time.
- **full_ft PushT vs the ResNet-18 specialist: not distinguishable.** full_ft has 22.7 ± 3.1%
  (34/150, Wilson [16.7, 30.0]) and ResNet-18 has 14.7 ± 6.1% (22/150, [9.9, 21.2]). The gap is
  +8.0 pp [−3.3, +16.0], and the intervals overlap. So the fine-tuned VLA closes the PushT gap that
  the frozen VLA had (finding 3), but it does not beat ResNet-18. On FetchPush, full_ft − ResNet-18 is
  +2.0 pp [−14.7, +18.7]: not distinguishable either. The comparison is not budget-matched: ResNet-18
  trained 4,000 steps at batch 128 on one robot, the VLA 2,500 steps at batch 64 shared by both robots.
- **The training budget matters as much as the encoder.** frozen_matched (2,500 steps × 64 = 160k
  samples) scores 24.0% mean, against 37.7% for the same frozen sep_heads model at 4,000 steps × 256
  (1.0M samples). On PushT, full_ft at the small budget (22.7%) beats frozen sep_heads at the large
  one (7.3%). That comparison mixes budget and encoder, so it is an observation, not a measured gap.

---

## What would change these numbers

- **Per-seed results.** Download the Kaggle output folders (each run's `metrics.json`) into
  `outputs/kaggle/` and rerun `.venv/bin/python scripts/aggregate.py`. The "approx." intervals then
  become exact paired bootstraps over the 50 start seeds. The FetchPush "unsolved starts" ranges then
  become exact counts, and the provisional gaps in findings 2, 3 and 5 are either confirmed or dropped.
  The files to download are the run folders under `/kaggle/working/outputs/` (for example
  `sep_heads-lora16-s2-bd071b5/metrics.json`), from each batch notebook's Output tab. The `.ipynb`
  notebooks themselves are not enough.
  Runs from now on also save per-seed results to W&B (`eval/<robot>_episode_success`).
- **The adaptation gaps.** With per-seed data, check the provisional FetchPush and mean gaps in finding 5.
  The two "real" PushT gaps are expected to stay real, given how far apart the seeds are.
- The bootstrap treats each training seed as fixed: it measures luck in the start states, not luck in
  training. The mean ± std columns show training luck, and it is large (sep_heads PushT ranges 0–18%
  across seeds).
