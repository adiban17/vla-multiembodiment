# Multi-Embodiment VLA: Build Plan & Architecture

Nov 24, 2026 · @Aditya

## Read this first

Yes, Claude Code can write all of it, and two days is enough. The binding constraint is about 11 GPU-hours, not code. So the GPU has to be lined up tonight and experiments must run overnight.

### Can Claude Code build it?

- **Yes.** The code is roughly 1,500 to 2,500 lines of standard PyTorch: data loaders, a small transformer, a training loop and a rollout evaluator. Nothing here needs novel engineering.
- **Where it needs you:** running things on a GPU, catching silent bugs (wrong image preprocessing, wrong action scaling), and judging whether results make sense. Each phase in the roadmap ends in a gate for exactly this.
- **Best setup:** run Claude Code directly on a rented cloud GPU machine. It can then run, debug and launch training itself instead of you pasting errors back and forth.
- **Second best:** Claude Code on your laptop writes the repo, you push to GitHub, and you run it in a Kaggle or Colab notebook with a T4. It works, but the debug loop is slower.

### The catch: the numbers on your resume

Your resume reports results from experiments that have not run yet. Building the project makes "Built a unified VLA policy" true. It does not make 68%, +6.4%, 94% or 38% true, and the real runs will give different numbers.

- **68% on PushT is a red flag on its own.** LeRobot's own Diffusion Policy, a strong documented open-source baseline, reaches 65.4% over 500 episodes after 175k training steps ([model card](https://huggingface.co/lerobot/diffusion_pusht)). A frozen-encoder regression head trained in under 45 minutes on a T4 will very likely land well below that. A VLA interviewer may know this reference number.
- **Small gaps can be noise.** With 150 rollouts per robot per configuration, a success rate's standard error is up to about 4 percentage points. A +4.1 gap may not be real, and research interviewers ask about this.
- **The plan:** build it, measure it, then replace every resume number with the measured one (section 14). Send the updated resume to the recruiter before the interview, and present only numbers you can open in W&B.

This is also the strongest interview position. "I got 31% on PushT; here is why it sits below Diffusion Policy and what I would try next" beats defending a number you cannot reproduce.

## The project in one minute

We train one neural network that controls two different robots by looking at a camera, reading an instruction and copying expert demonstrations. Then we test three ways of telling that network which robot it is driving.

**The black box**

- **In:** the last 2 camera frames, the robot's own state (where its hand is), one sentence ("push the T onto the target"), and which robot this is.
- **Out:** the next 16 actions, in that robot's own action format.
- **Robot 1, PushT:** a circular pusher on a flat 2-D table. Each action is 2 numbers: where to move next (x, y). Goal: push a T-shaped block onto a T-shaped outline.
- **Robot 2, FetchPush:** a simulated 7-joint robot arm. Each action is 4 numbers: move the hand by dx, dy, dz, plus a gripper command. Goal: push a cube onto a red target.

**How it learns:** imitation, not reinforcement learning. We have recorded expert demonstrations: humans for PushT, a script we write for FetchPush. The network sees a moment from a demo and is trained to output what the expert did next. No rewards are involved.

**What is inside the box**

1. **Eyes:** DINOv2-small, a pretrained vision model, turns each image into a set of 384-number vectors. Frozen: we never train it in the main model.
2. **Ears:** MiniLM, a pretrained sentence model, turns the instruction into one 384-number vector. Also frozen.
3. **Brain:** a small transformer we train from scratch (about 15M parameters). It reads all those vectors and writes out the actions.

**The research questions, which are the actual point**

1. One brain, two bodies with different action sizes (2 and 4). How should the brain know which body it controls? (A) a learned robot-ID token, (B) FiLM, where the robot ID rescales features inside every layer, or (C) a shared brain with a separate output head per robot.
2. If we let the eyes adapt, does cheap LoRA fine-tuning match full fine-tuning, and how much GPU memory does it save?
3. Does the shared model beat simpler per-robot policies: an MLP trained from scratch and a ResNet-18?

**Why Miko will probe this hardest:** their JD asks for almost exactly this project. It names "two robotic embodiments with different action dimensionalities" and "conditioning strategies to support both within a unified model". It also asks for full and parameter-efficient fine-tuning, action branches built from scratch, structured ablations, and seed and config tracking.

## Concepts from first principles

The whole project rests on the ideas below. For each one: what goes in, what comes out, why we need it, and the sentence to say if an interviewer asks.

### Learning to act

- **Behavior cloning (BC).** In: pairs of (observation, expert action). Out: a policy mapping observation to action. It is plain supervised regression from what the robot sees to what the expert did.
  - Weakness: covariate shift. A small mistake puts the robot somewhere the expert never went, and errors compound.
  - Say: "BC is supervised learning on demonstrations; its failure mode is compounding error from covariate shift."
- **Rollout.** Run the policy in the simulator from a fresh start until success or a time limit. Training loss is measured on demos; success is measured by rollouts. They can disagree, and only rollouts count.
- **Action chunking.** In: one observation. Out: the next 16 actions at once. We execute the first 8, look again, and re-plan.
  - Why: fewer decisions means less compounding error, and motion is smoother. The idea comes from ACT (Zhao et al., 2023).
  - Say: "Chunking shortens the effective decision horizon and gives temporal consistency."
- **Multimodality.** There are several valid ways to push the T, for example around the left or around the right. A regression loss averages them into a move between two good options, which is itself bad.
  - Diffusion Policy and flow-matching heads (as in π0) model the whole distribution of actions to fix this.
  - Our head uses L1 regression. This is the main reason PushT will be hard for us. Say it before they do.

### Bodies and worlds

- **Embodiment.** One specific robot body: its sensors, joints and action space. Multi-embodiment means one policy for several bodies, as in Open X-Embodiment, Octo and π0.
- **Action space.** The numbers a policy outputs every step, and what they mean.
  - PushT: 2 numbers, the absolute target position for the pusher, in pixels from 0 to 512.
  - FetchPush: 4 numbers in \[-1, 1\]: dx, dy, dz of the hand, plus a gripper command.
  - Detail worth knowing: in FetchPush the gripper is locked closed, so the 4th number has no effect.
- **PushT** (gym-pusht, from the Hugging Face LeRobot team). 2-D physics using pymunk, not MuJoCo. Observation: a 96×96 RGB image plus the pusher's (x, y). Success: the T covers at least 95% of the target at any point. LeRobot evaluates up to 300 steps. Data: `lerobot/pusht`, about 200 human demonstrations.
- **FetchPush** (Gymnasium-Robotics). 3-D physics in MuJoCo, a Fetch arm, 25 Hz control, 50 steps per episode. Success: cube within 5 cm of the target. There is no standard image demo set, so we script an expert and record our own.
- **MuJoCo.** A physics simulator for contacts, joints and friction. We use it only through FetchPush; it renders images on a GPU server without a screen through EGL.

### Seeing and reading

- **DINOv2-small.** In: a 224×224 image. Out: one summary vector (CLS) plus 256 patch vectors on a 16×16 grid, each 384 numbers. A self-supervised ViT-S/14 from Meta with about 22M parameters.
  - We average-pool the 16×16 grid to 8×8, giving 64 patch tokens plus CLS: 65 tokens per frame.
  - Why not CLIP: DINOv2 features keep spatial detail (where things are), which control needs. CLIP features are tuned for matching images to captions. OpenVLA fuses DINOv2 with SigLIP for this reason.
- **MiniLM** (`all-MiniLM-L6-v2`). In: a sentence. Out: one 384-number vector, the mean of its token embeddings. A 6-layer distilled sentence encoder with about 22M parameters.
  - Honest note: with only 2 fixed instructions, language acts like a task ID. We train on paraphrases and test on held-out ones to show it is not brittle.
- **Frozen encoder.** Weights fixed, no gradients.
  - Pros: we can compute features once and cache them, so training is fast; a 22M model cannot overfit 200 demos; little memory.
  - Con: features may miss task-specific detail. The adaptation ablation measures that cost.

### The brain, and how it learns which body it drives

- **Transformer action head.** In: a sequence of tokens (vision, language, state, plus 16 learned action-query tokens). Out: one vector per query, mapped to an action. Every token attends to every other token.
  - The queries are blank slots the transformer fills with "what to do at step k". ACT's decoder queries and Octo's readout tokens work the same way.
- **(A) Learned embodiment token.** One extra learnable 384-number vector per robot, added to the sequence. The transformer has to learn to attend to it, and it can get diluted among about 150 tokens.
- **(B) FiLM, Feature-wise Linear Modulation** (Perez et al., 2018). In: the robot ID. Out: a scale γ and a shift β for every channel in every layer. The robot's identity touches every layer and cannot be ignored. γ and β start at zero, so training starts from the unconditioned model. Diffusion transformers use the same trick (adaLN).

```latex
h \leftarrow (1 + \gamma_{\ell}(e)) \odot \mathrm{LayerNorm}(h) + \beta_{\ell}(e)
```

- **(C) Separate heads.** One shared trunk, and a different final output layer per robot (2 numbers or 4). Identity enters only at the very end, so the trunk must infer which robot it is from the image.
- **Padding and masking.** To share one output layer, PushT actions are padded from 2 to 4 numbers and the loss ignores the padded ones. At run time we keep only the first 2. Octo and π0 handle mixed action spaces the same way.
- **Normalization.** Each robot's actions and states are rescaled to \[-1, 1\] using that robot's own data. Without it, PushT's 0 to 512 pixel actions would swamp FetchPush's ±1. It is the most common source of silent bugs.

### Adapting big models cheaply

- **Full fine-tuning.** Unfreeze DINOv2 and train all 22M weights with a small learning rate. The most flexible option; it needs gradients and Adam state for every weight.
- **LoRA, rank 16.** Keep each weight matrix W frozen and learn a low-rank update ΔW = B·A, with A of size 16×384 and B of size 384×16. We apply it to the query, key and value layers in all 12 DINOv2 blocks: about 0.44M trainable numbers instead of 22M.
  - Memory is saved on gradients and optimizer state (Adam stores 2 extra numbers per trainable weight). Activations must still be stored for backpropagation, so the saving is partial.
  - Say: "LoRA's memory saving comes from gradients and optimizer state, not activations."

```latex
W' = W + \frac{\alpha}{r} B A, \quad A \in \mathbb{R}^{r \times d},\; B \in \mathbb{R}^{d \times r},\; r = 16,\; \alpha = 32
```

### Measuring honestly

- **Success rate.** The fraction of rollouts that meet the success condition. We run 50 rollouts per seed × 3 seeds = 150 per configuration per robot.
- **Seeds and error bars.** A seed fixes the random initialization and data order. Different seeds give different results, so we report mean ± standard deviation across 3 seeds.
  - With 150 rollouts, a 50% success rate has a standard error of about 4 points: √(0.25/150) ≈ 0.041. Treat gaps under about 8 points with caution.
- **Percentage points, not percent.** 60% to 66% is +6 percentage points (pp), or +10% relative. Always say "pp" for differences in success rates.

## System architecture

One transformer trunk is shared by both robots. The three conditioning variants differ only in where the robot's identity enters: as a token (A), inside every layer (B), or at the output head (C).

&#91;embedded content: VLA policy architecture · inputs, encoders, token sequence, trunk, heads\]

Read it top to bottom: four inputs become tokens, the trunk mixes them, and the 16 query slots come out as one action chunk. Everything except the dashed paths is identical across variants.

### One sample, step by step

A training batch adds a leading batch dimension B to every shape below.

| Step | Shape for one sample | Trained? |
| --- | --- | --- |
| Images in | 2 × 3 × 224 × 224 | n/a |
| DINOv2-small output | 2 × 257 × 384 (CLS + 16×16 patches) | frozen |
| Pool patches 16×16 to 8×8, keep CLS | 2 × 65 × 384, flattened to 130 × 384 | no parameters |
| Vision projection, Linear 384 to 384 | 130 × 384 | yes |
| MiniLM sentence vector, then Linear | 1 × 384 | encoder frozen, projection trained |
| State (8 numbers), Linear 8 to 384 | 1 × 384 | yes |
| Embodiment token, variant A only | 1 × 384 | yes |
| Action queries | 16 × 384 | yes |
| Concatenate, add position and token-type embeddings | 149 × 384 (148 without A) | yes |
| Trunk: 8 layers | 149 × 384 | yes |
| Take the 16 query outputs, then the head MLP | 16 × 4 shared, or 16 × 2 / 16 × 4 separate | yes |
| Un-normalize, keep the robot's real dimensions | 16 × 2 for PushT, 16 × 4 for FetchPush | n/a |

At run time the same path runs once per re-plan, every 8 environment steps, with DINOv2 and MiniLM computed live instead of read from the cache.

## Data pipeline

Both robots end up in one sample format, normalized per robot. Frozen encoder features are cached to disk once, so frozen-encoder training never runs DINOv2 or MiniLM.

### PushT: download

1. Load `lerobot/pusht` with `LeRobotDataset`. Current releases import it from `lerobot.datasets.lerobot_dataset`; pin the version you install, because this path has moved before.
2. Fields used: `observation.image` (96×96 RGB), `observation.state` (pusher x, y) and `action` (target x, y). About 200 episodes at 10 Hz.
3. Decode every video frame once into a uint8 NumPy array. LeRobot stores images as MP4, and decoding inside the training loop is slow.
4. Split by episode: 90% train, 10% held out for validation loss. Validation episodes are never used for rollouts.

### FetchPush: generate

1. Environment: `FetchPush-v4` with `render_mode="rgb_array"`, a 128×128 camera and `MUJOCO_GL=egl`.
2. Scripted expert, closed-loop (it re-reads the cube's position every step):
   1. d = unit vector from cube to goal, in the table plane.
   2. Move the hand above a point 6 cm behind the cube, on the side opposite d.
   3. Lower the hand to cube height.
   4. Push along d until the cube is within 2 cm of the goal.
   5. Action = clip(K · (target − hand), −1, 1). Gripper action fixed at 0. Add small Gaussian noise so the demos vary.
3. Collect 500 successful episodes × 50 steps, about 25k frames. That is close to PushT's size, so neither robot dominates. Discard failures, and log the expert's raw success rate (target: 80% or more).
4. Save per step: the image, state = hand xyz (3) + finger widths (2) + goal xyz (3), the 4-number action, and the env seed.
5. **Design decision:** the goal position is given as numbers in the state, but the cube's position must come from pixels. This keeps the vision problem real while solvable in 2 days. Say so if asked.

### One sample, both robots

| Field | Shape | PushT | FetchPush |
| --- | --- | --- | --- |
| images | 2 × 3 × 224 × 224 | 96 px frames, upscaled | 128 px frames, upscaled |
| state | 8 | 2 real + 6 zeros | 8 real |
| instruction | text | 1 of 8 training paraphrases | 1 of 8 training paraphrases |
| embodiment\_id | 1 | 0 | 1 |
| actions | 16 × 4 | 2 real + 2 padded per step | 4 real |
| action\_dim\_mask | 4 | 1, 1, 0, 0 | 1, 1, 1, 1 |
| action\_is\_pad | 16 | true past episode end | true past episode end |

The 2 frames are t−1 and t. At the first step of an episode, frame t is repeated.

### Normalization

- Per robot and per dimension, min-max scale states and actions to \[−1, 1\] using the train split. Save to `stats.json` and load the same file at evaluation.
- FetchPush actions are already in \[−1, 1\] and are left as they are.
- One function, `preprocess_image()`, is used for caching, training and evaluation: bicubic resize to 224, then ImageNet mean and std.

### Instructions

- Write 10 paraphrases per task, such as "push the T-shaped block onto the target" and "slide the T into the outline".
- Train on 8, hold out 2 for a robustness check. MiniLM embeddings for all 20 sentences are computed once.

### Feature cache: the speed trick

- Run DINOv2-small once over every frame and store CLS + 8×8 pooled patches = 65 × 384 numbers per frame, in fp16.
- About 50k frames × 65 × 384 × 2 bytes ≈ 2.5 GB. Store it as a memory-mapped file.
- Frozen runs read the cache. Adaptation runs (full fine-tuning and LoRA) cannot, and read raw images instead.
- Test: features computed live during evaluation must match the cached features for the same frame, within fp16 tolerance.

## Ablations: what changes in each run

Every ablation run shares the data, trunk size, loss, step count and evaluation. Only the one factor named in each table changes, so any difference in success can be pinned on that factor.

### Conditioning variants (encoders frozen)

| Variant | How robot identity enters | Output head | Extra parameters |
| --- | --- | --- | --- |
| None (optional control) | Not given; the trunk must infer it from image and language | Shared MLP, 4 outputs, padded dims masked | 0 |
| A. Learned token | One learned 384-d vector per robot, added to the sequence | Shared MLP, 4 outputs, padded dims masked | 768 |
| B. FiLM | 64-d robot embedding → γ and β for both LayerNorms in all 8 layers, zero-initialized | Shared MLP, 4 outputs, padded dims masked | about 0.8M |
| C. Separate heads | Only through which head is used | One MLP per robot: 2 outputs or 4 | about 0.15M |

Every head is a 2-layer MLP (384 → 384 → outputs), so C differs from A only in having two of them.

A likely interview question: "Is FiLM winning only because it has about 5% more parameters?" If FiLM wins by a lot, run one control: variant A with a 9th trunk layer, which roughly matches FiLM's parameter count.

### Adaptation regimes (on the best conditioning variant)

The best variant is chosen from the frozen runs by mean success across both robots. Write that rule down before the results arrive.

| Regime | DINOv2 parameters trained | Learning rate (encoder / head) | What we record |
| --- | --- | --- | --- |
| Frozen (reference) | 0 | none / 3e-4 | success, minutes per run |
| Full fine-tuning | all, about 22M | 1e-5 / 3e-4 | success, peak VRAM, minutes per run |
| LoRA r=16, α=32 | about 0.44M (query, key, value in 12 blocks) | 1e-4 / 3e-4 | success, peak VRAM, minutes per run |

- MiniLM stays frozen in every regime. With 20 fixed sentences there is nothing to adapt, and gradients through it would only overfit.
- Fair memory comparison: same batch size and the same mixed precision; read `torch.cuda.max_memory_allocated()` after 200 steps.
- "LoRA retains X% of full fine-tuning" = LoRA success ÷ full fine-tuning success, averaged over the two robots.

### Cheap extras, only if ahead of schedule

1. Held-out paraphrases: does success drop with the 2 unseen instructions?
2. FiLM trained on one robot at a time: does joint training help or hurt each robot?
3. History of 1 frame instead of 2.

## Baselines

Two per-robot specialists show what the pretrained encoders and the shared transformer buy us. Both use the same action chunks, loss, normalization and evaluation, so only the model differs.

| Baseline | Vision | Trained on | Parameters (approx.) | What it isolates |
| --- | --- | --- | --- | --- |
| BC-MLP, from scratch | None: flattened 64×64 pixels of 2 frames + state, into a 3-layer MLP (512 wide) | Each robot separately | 12.9M | The value of any pretrained vision at all |
| ResNet-18 BC | ImageNet-pretrained ResNet-18, fine-tuned end to end; 512-d pooled feature per frame + state, into an MLP | Each robot separately | 11.8M | Pretrained CNN vs frozen DINOv2 + shared transformer |

- ResNet-18 is the vision encoder Diffusion Policy uses, which makes it a natural baseline. Diffusion Policy also swaps BatchNorm for GroupNorm; that swap is optional here.
- **External reference, not re-run:** LeRobot's Diffusion Policy reaches 65.4% PushT success over 500 episodes ([model card](https://huggingface.co/lerobot/diffusion_pusht)). Quote it as context for your PushT number, not as a baseline you beat.
- **One caveat to state yourself:** the baselines are specialists, while our model is one network for both robots. If ours wins, the gain mixes "better encoder" with "joint training". The optional single-robot FiLM run separates the two.

## Training recipe

One config file controls every run. The defaults below are a starting point, tuned only on validation loss and the Phase 3 smoke run, never on final evaluation rollouts.

### Model sizes

- Trunk: 8 transformer layers, width 384, 6 attention heads, feed-forward width 1536, pre-LayerNorm, GELU, dropout 0.1, no causal mask.
- Sequence per sample: 1 embodiment token (variant A only) + 1 language + 1 state + 130 vision (65 × 2 frames) + 16 action queries = up to 149 tokens.
- Expected trainable parameters: about 14.7M for variant A and about 15.5M for FiLM. The number on the resume is whatever `count_params()` prints.

### Optimization

| Setting | Frozen runs | Adaptation runs | Baselines |
| --- | --- | --- | --- |
| Optimizer | AdamW, weight decay 0.01 | same | same |
| Learning rate | 3e-4 | head 3e-4; DINOv2 1e-5 (full) or 1e-4 (LoRA) | 1e-3 (MLP), 1e-4 (ResNet-18) |
| Schedule | 500 warmup steps, cosine decay to 0 | same | same |
| Batch size | 256 | 64 | 128 |
| Steps | 20k | about 8k, whatever fits 45 minutes | 20k |
| Precision | fp16 autocast + GradScaler | same | same |
| Gradient clipping | 1.0 | same | same |
| Batch mix | 50% PushT, 50% FetchPush | same | one robot |

- **Why fp16, not bf16:** the T4 is a Turing GPU without native bf16, so we use fp16 with loss scaling.
- **Loss:** L1 over real action dimensions and real time steps only. m is 1 where the step is not padding and the dimension is real, else 0.

```latex
\mathcal{L} = \frac{\sum_{k=1}^{16} \sum_{j=1}^{4} m_{kj}\, \lvert \hat{a}_{kj} - a_{kj} \rvert}{\sum_{k,j} m_{kj}}
```

### Seeds and checkpoints

- One integer seeds Python, NumPy, PyTorch, CUDA and the data-loader workers. Seeds are 0, 1 and 2.
- Log validation L1 every 1,000 steps. Evaluate the final checkpoint only; picking the best checkpoint by rollouts would inflate results.
- Expect validation loss and rollout success to disagree sometimes. That is normal for behavior cloning and a good thing to mention.

## Evaluation protocol

Every trained run is rolled out 50 times per robot from the same 50 fixed starting states, so every model faces identical test conditions.

1. **Fixed starts.** Environment seeds 100000 to 100049 for both robots, shared by every run. This makes comparisons paired: model X and model Y face the same 50 situations.
2. **The control loop.** Observe → `preprocess_image()` (the same function as training) → DINOv2 and MiniLM run live → the policy predicts 16 actions → un-normalize → execute the first 8 → repeat.
3. **PushT.** Up to 300 steps. Success if coverage reaches 95% at any step. Also log maximum coverage (0 to 1), a smoother secondary metric that still moves when success is near 0.
4. **FetchPush.** 50 steps. Success if `info["is_success"]` is true at the final step, meaning the cube is within 5 cm of the target.
5. **Numbers per run:** success rate over its 50 rollouts. **Per configuration:** mean ± standard deviation over 3 seeds, plus a 95% Wilson interval over the pooled 150 rollouts.
6. **Headline "mean success":** the average of the PushT and FetchPush rates.
7. **Calling a winner:** report gaps in pp with intervals. Treat gaps under about 8 pp as not distinguishable unless a paired bootstrap over the 150 starting states says otherwise.
8. **Videos:** save 3 rollouts per run to W&B: one success, one failure and one random.

### Sanity checks before trusting any number

- Replay a FetchPush demo's saved actions from its saved env seed: it should succeed.
- Run the scripted FetchPush expert through the evaluation harness: success should match its known rate.
- A random policy should score about 0% on both robots.
- A model with low validation loss but about 0% success almost always means a preprocessing or normalization mismatch between training and evaluation.

## Experiment matrix and compute budget

The required matrix is 28 runs and about 11 T4 GPU-hours. Frozen-encoder runs are cheap; the adaptation runs are the expensive part. Minutes per run are estimates until the Phase 3 smoke test measures steps per second.

| Priority | Group | Configurations | Seeds | Runs | Est. minutes per run, incl. eval | Est. GPU-hours |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | Feature cache | DINOv2 + MiniLM over all frames | none | 1 | 30 | 0.5 |
| P0 | Conditioning, frozen | Token, FiLM, separate heads | 3 | 9 | 20 | 3.0 |
| P1 | ResNet-18 BC | PushT, FetchPush | 3 | 6 | 25 | 2.5 |
| P1 | Adaptation | Full fine-tuning, LoRA r=16 | 3 | 6 | 45 | 4.5 |
| P1 | BC-MLP | PushT, FetchPush | 3 | 6 | 8 | 0.8 |
| P2 | Optional controls | No conditioning; single-robot FiLM; matched-parameter token | 3 | 6 to 9 | 20 | 2 to 3 |

If time or GPU quota runs short, cut from the bottom: P2 first, then the second and third seeds of the adaptation runs. Report any reduced seed count plainly.

### Where to run

1. **Recommended:** a rented cloud GPU machine with Claude Code installed on it. Claude Code can then launch, watch and fix runs itself.
2. **Free option:** Kaggle notebooks offer T4 GPUs under a weekly quota (check the current limit). Two T4s let two seeds run in parallel.
3. **Colab** works too, but free sessions can disconnect. Write checkpoints and results to Drive or W&B as you go.

- If you run on a faster GPU than a T4, the "<45 min/run T4" line on the resume must change to the GPU you actually used.
- Every run writes its metrics to W&B and a local file the moment it finishes, so a dropped session costs one run, not a day.

## Repo layout and experiment hygiene

Every number on the resume must trace back to one run folder holding its config, git commit, seed and metrics.

```
vla-multiembodiment/
├── CLAUDE.md                 # project brief and rules for Claude Code
├── README.md                 # results table, how to reproduce
├── requirements.txt          # pinned versions
├── docs/spec.md              # sections 4 to 11 of this doc
├── configs/
│   ├── base.yaml
│   ├── cond/      none.yaml, token.yaml, film.yaml, sep_heads.yaml
│   ├── adapt/     frozen.yaml, full_ft.yaml, lora16.yaml
│   └── baseline/  bc_mlp.yaml, resnet18.yaml
├── src/vla/
│   ├── envs/      pusht.py, fetch.py            # make_env, render, success
│   ├── data/      pusht_loader.py, fetch_expert.py, collect_fetch.py,
│   │              dataset.py, normalize.py, instructions.py, cache_features.py
│   ├── models/    encoders.py, trunk.py, conditioning.py, policy.py, baselines.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils/     config.py, seed.py, tracking.py
├── scripts/       sweep.sh, aggregate.py, plots.py
├── tests/         test_shapes.py, test_overfit.py, test_expert.py, test_cache.py
└── runs/<run_id>/ config.yaml, git.txt, metrics.json, ckpt.pt, videos/
```

- **Configs:** YAML files merged with OmegaConf, with command-line overrides such as `cond=film seed=1`. The fully resolved config is saved in the run folder and in W&B.
- **Git:** a sweep run refuses to launch with uncommitted changes and records the commit hash. A `--allow-dirty` flag exists for debugging and saves the `git diff` alongside.
- **Run IDs:** `{cond}-{adapt}-s{seed}-{commit}`, for example `film-frozen-s1-3f9a2c1`.
- **W&B:** one project, grouped by conditioning, adaptation and baseline. Log train and validation loss, learning rate, steps per second, peak VRAM, success per robot and rollout videos.
- **Results:** `scripts/aggregate.py` collects every `metrics.json` into one CSV and the README table. Resume numbers are copied from that CSV and nowhere else.
- **Determinism:** seeds are set everywhere, but GPU kernels are not bit-exact, so reruns match closely rather than exactly. That is a fine answer if asked.

## Two-day roadmap

Nine phases, about 18 hours of hands-on work plus GPU time overnight. Each phase ends in a gate you check yourself before moving on. Each also lists what you must be able to explain out loud before it counts as done.

### Tonight

1. **Phase 0: setup (1.5 h)**
   - Build: rent or open a GPU machine and install Claude Code on it. Create W&B and Hugging Face accounts and a new GitHub repo. Save sections 4 to 11 of this doc as `docs/spec.md`.
   - Gate: `nvidia-smi` shows the GPU, and `torch.cuda.is_available()` prints True.
   - Learn: sections 2 and 3 of this doc. Then explain the black box out loud in 60 seconds, without notes.

### Day 1

2. **Phase 1: environments and data (2.5 h)**
   - Build: PushT loader (LeRobot to NumPy), gym-pusht wrapper, FetchPush with EGL rendering, scripted expert, 500 FetchPush demos.
   - Gate: a 3-episode video per robot; expert success of 80% or more; dataset stats printed and plausible (PushT actions between 0 and 512).
   - Learn: what every number in each robot's state and action means; why the expert is closed-loop.
3. **Phase 2: unified dataset and feature cache (1.5 h)**
   - Build: normalization stats, windowed dataset (2 frames, 16-action chunks, masks), instructions and paraphrases, DINOv2 and MiniLM cache.
   - Gate: shape tests pass; un-normalizing a normalized action returns the original; cached and live features match; cache is about 2.5 GB.
   - Learn: padding and masking; why normalization is per robot; what the 65 tokens per frame are.
4. **Phase 3: model and training loop (2.5 h)**
   - Build: trunk, the conditioning variants, policy, `train.py` with OmegaConf, seeds, git hash, W&B and mixed precision.
   - Gate: every variant overfits a single batch to near-zero loss; parameter counts printed; a 1,000-step smoke run logs to W\&B; steps per second measured and the compute table updated.
   - Learn: trace one sample's tensor shapes through the model; explain the FiLM line in the code.
5. **Phase 4: evaluation harness (2 h)**
   - Build: `evaluate.py` with chunked execution, success metrics, Wilson intervals and videos.
   - Gate: the scripted expert scores its known rate through the harness; a random policy scores about 0%; a 5k-step FiLM model gets non-zero FetchPush success.
   - Learn: closed-loop vs open-loop control; why evaluation preprocessing must match training.
   - Before bed: commit, then launch the 9 conditioning runs overnight.

### Day 2

6. **Phase 5: baselines (2 h)**
   - Build: BC-MLP and ResNet-18 on the same data and evaluation; launch 12 runs.
   - Gate: both overfit one batch; the first run's evaluation completes.
   - Learn: what each baseline isolates.
7. **Phase 6: adaptation regimes (2 h)**
   - Build: the un-frozen DINOv2 path on raw images, full fine-tuning, LoRA with the PEFT library (r=16, α=32, query/key/value), VRAM and time logging. Launch 6 runs on the best conditioning variant.
   - Gate: trainable counts printed (about 0.44M LoRA parameters in the encoder); peak VRAM logged; one run measured under 45 minutes.
   - Learn: where LoRA saves memory and where it does not.
8. **Phase 7: results (2 h)**
   - Build: `aggregate.py` producing a results CSV and README tables; two plots (success by conditioning with error bars, success against peak VRAM for adaptation); rollout GIFs.
   - Gate: every README number traces to a run ID.
   - Learn: write 5 findings in your own words, including at least 2 failure modes you saw in the videos.
9. **Phase 8: resume and interview (2 h)**
   - Rewrite the resume bullets with measured numbers (section 14) and send the update to the recruiter.
   - Drill section 15 out loud. I will quiz you on it.

### Prompts for Claude Code

Start every phase in a fresh Claude Code session with the same pattern. It writes the gate checks first, so "done" means something.

Phase 0:

```
Read docs/spec.md. Write CLAUDE.md summarizing the architecture, repo layout and these rules:
pinned dependencies; every run saves its resolved config and git hash; a single
preprocess_image() shared by caching, training and evaluation; tests before features.
Then create the repo skeleton and requirements.txt. Do not implement models yet.
```

Phase 1:

```
Implement Phase 1 from docs/spec.md: src/vla/envs and src/vla/data (pusht_loader,
fetch_expert, collect_fetch). Write tests/test_expert.py asserting at least 80% expert
success over 50 seeds. Save 3 sample videos per robot to runs/debug/. When done, stop and
report: episodes, frames, action min/max per dimension, expert success rate.
```

Phases 2 to 7:

```
Implement Phase <N> from docs/spec.md. First write the gate checks listed for this phase
as tests or scripts. Then implement until they pass. Run them, show me the output,
and stop. Do not start the next phase.
```

After each phase, paste the gate output here and I will check it with you before you move on.

## Risks and fallbacks

The most likely problem is low PushT success; the most dangerous is a silent train/eval mismatch. Each risk below has a fallback decided in advance.

| Risk | Likelihood | How it shows up | Fallback |
| --- | --- | --- | --- |
| PushT success very low (under 20%) with an L1 head | High | Rollouts | Report max coverage as well; cache 2 to 3 randomly shifted crops per frame for augmentation; train longer. A flow-matching head is a stretch goal only if a day ahead. |
| Train/eval preprocessing mismatch | Medium, silent | Low validation loss, about 0% success | `test_cache.py`; one `preprocess_image()` everywhere; check stats.json is loaded at eval |
| MuJoCo will not render without a screen | Medium | EGL error at startup | `MUJOCO_GL=osmesa` (slower, CPU), or render demos on another machine and upload |
| LeRobot version or API change | Medium | Import or decode errors | Pin the version; fall back to reading the dataset's parquet and MP4 files directly |
| Scripted FetchPush expert below 80% | Medium | `test_expert.py` fails | Tune the gain and the behind-cube offset; re-approach if the cube rotates; keep successes only |
| GPU time or quota runs out | Medium | Runs queue up | Follow the priority order in section 10; report reduced seed counts plainly |
| FiLM wins only through extra parameters | Low to medium | Asked in the interview | Matched-parameter token control (one extra trunk layer) |
| Results differ from the resume | Certain | Phase 7 | Section 14 |

## Resume, line by line

The architecture claims will be true once we build; every performance number is a placeholder until Phase 7, then gets replaced by a measured value or removed.

| Resume claim | Status after the build | Where the real value comes from |
| --- | --- | --- |
| Frozen DINOv2-small + MiniLM with a lightweight transformer action head | True | The code |
| 14.8M trainable parameters | Expected 14.7M to 15.5M depending on variant | `count_params()` log for the main model |
| Jointly trained on PushT (2-D) and FetchPush (4-D) | True; know that FetchPush's 4th dimension is inert | The code |
| <45 min/run on a T4 | Likely true for frozen runs if measured on a T4; check adaptation runs | W&B runtime |
| FiLM +6.4 pp over token, +4.1 pp over separate heads | Unknown; may fall within noise | Results CSV with intervals |
| LoRA retains 94% of full fine-tuning at 38% lower VRAM | Unknown | Adaptation runs |
| 68% PushT, 74% FetchPush (3 seeds × 50 rollouts) | 68% would beat LeRobot's Diffusion Policy (65.4%); expect much lower | Results CSV |
| BC-MLP 41% / 48%, ResNet-18 56% / 63%, gains of +26.5 and +11.5 | Unknown | Results CSV |
| Per-run YAML configs, git hash logging, W&B tracking | True after Phase 3 | The repo |

### Rewrite template, filled in from the results CSV

- Built a unified VLA policy pairing frozen DINOv2-small and MiniLM encoders with a \[X.X\]M-parameter transformer action head, jointly trained on PushT (2-D) and FetchPush (4-D) action spaces in under \[N\] min/run on a T4.
- Ablated embodiment conditioning (learned token, FiLM, separate heads) and encoder adaptation (full fine-tuning vs LoRA r=16): \[measured finding, with pp gaps and whether they are significant\]; LoRA reached \[Y\]% of full fine-tuning success at \[Z\]% lower peak VRAM.
- Reached \[A\]% / \[B\]% success on PushT / FetchPush (3 seeds × 50 rollouts) vs \[C\]% / \[D\]% for a ResNet-18 BC baseline and \[E\]% / \[F\]% for a scratch BC-MLP, with per-run YAML configs, git-hash logging and W&B tracking.

If a gap is not significant, say so on the resume: "differences within ±X pp" reads as rigor to a research team. If a baseline wins on one robot, write that too, and have the explanation ready.

The second project (latent world model with CEM planning) has the same issue. We will handle it in its own session.

## Interview drill

These 15 questions are the most likely ones about this project. Answer each out loud in under a minute; answers marked \[results\] get filled in after Phase 7.

1. **Walk me through the architecture.** Two frames go through frozen DINOv2-small: 65 tokens each. The instruction goes through frozen MiniLM: 1 token. State goes through a linear layer: 1 token. Add 16 learned action queries; an 8-layer transformer reads all \~149 tokens. Each query becomes one future action. Robot identity enters as a token, through FiLM, or through separate heads.
2. **Why freeze the encoders?** Speed, since features are cached once. Data size, since about 200 PushT demos would overfit a 22M-parameter encoder. Memory. I measured the cost of freezing with the full fine-tuning and LoRA ablation. \[results\]
3. **Why DINOv2 and not CLIP or SigLIP?** DINOv2's self-supervised features keep spatial and geometric detail, which manipulation needs. CLIP-style features are tuned to match captions. OpenVLA fuses DINOv2 with SigLIP for this reason.
4. **How do you handle 2-D and 4-D actions in one model?** Pad to 4 and mask the loss on padded dimensions; at run time, un-normalize and keep the first d. The separate-heads variant avoids padding by giving each robot its own output layer.
5. **Why might FiLM beat a token?** FiLM injects identity multiplicatively into every layer, so it cannot be ignored. A token relies on attention to find one vector among about 150. Then give the measured gap and whether it is significant. \[results\]
6. **What does language actually contribute?** Honestly, with one task per robot it works like a task ID, and it is confounded with embodiment. I tested held-out paraphrases. Language would matter with several tasks per robot.
7. **What is LoRA, why r=16, and where does it save memory?** A frozen weight plus a trainable low-rank update B·A. Rank 16 is a common middle ground; OpenVLA showed LoRA can match full fine-tuning on robot data. Memory is saved on gradients and Adam state, not activations. \[results\]
8. **Why action chunks?** They cut the number of decisions, which reduces compounding error, and give smoother motion. We re-plan every 8 steps to stay closed-loop.
9. **Why is PushT hard for your model?** Multimodality: an L1 head averages distinct valid pushes. Diffusion Policy reaches 65.4% by modeling the action distribution. My next step would be a flow-matching or diffusion head.
10. **How did you make comparisons fair and reproducible?** Same data, steps, loss and evaluation; 3 seeds; 50 fixed starting states shared by all models; final checkpoint only; resolved config and git hash saved per run.
11. **Are your differences significant?** 150 rollouts per robot gives a standard error up to about 4 pp. I report Wilson intervals and use a paired bootstrap over the shared starting states. \[results\]
12. **What is covariate shift, and how would you reduce it?** The policy drifts into states missing from the demos and errors compound. Fixes: DAgger (query the expert on the policy's own states), noise injection during demos (our expert adds noise, similar to DART), more data, chunking.
13. **How does this compare to OpenVLA, Octo and π0?** OpenVLA: a 7B Llama 2 model with fused DINOv2 and SigLIP vision; actions become 256-bin discrete tokens. Octo: a transformer with readout tokens and a diffusion action head, trained on many embodiments. π0: a PaliGemma vision-language model plus a flow-matching action expert producing action chunks. Mine is a miniature of the frozen-encoder plus action-head design.
14. **How would you add a world model?** Train a model that predicts the next DINOv2 features from current features and an action chunk. Use it as an auxiliary loss, or plan with it by scoring sampled action chunks. This links to my second project and to Miko's "World Action Models".
15. **What failed, and what would you do next?** Use the 5 findings from Phase 7. \[results\]

## Sources

- [lerobot/diffusion\_pusht model card](https://huggingface.co/lerobot/diffusion_pusht): 65.4% PushT success over 500 episodes; success = max overlap of at least 95%.
- [Gymnasium-Robotics FetchPush documentation](https://robotics.farama.org/envs/fetch/push/): 4-D action space, gripper locked closed, 0.05 m success threshold, 50-step episodes.
- [gym-pusht README](https://github.com/huggingface/gym-pusht): 2-D action as target position in 0 to 512, 96×96 image observations, 95% coverage success.
