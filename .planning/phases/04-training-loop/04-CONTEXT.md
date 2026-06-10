# Phase 4: Training Loop - Context

**Gathered:** 2026-06-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire all DDPG components into a complete, runnable training loop — environment, agent update step, 96 batched parallel environments, exploration noise annealing — and produce a training run that shows measurable reward improvement over the first 1000+ episodes.

Deliverables:
- `env.py` — `BatchedPaintEnv`: vectorized reset/step for 96 envs, sequential k=5 stroke compositing via frozen R, normalized L2 incremental reward
- `ddpg/agent.py` — `update_step()` implemented: critic Bellman loss, actor policy gradient through frozen R, soft target updates
- `train.py` — CLI-launchable training loop, TensorBoard logging, checkpoint saving/resuming, configurable arguments
- `prepare_data.py` — one-time extraction script for the HuggingFace dataset to disk

Out of scope: eval rollout, timelapse export, palette projection, WGAN (Phase 5).

</domain>

<decisions>
## Implementation Decisions

### Training Dataset

- **D-01:** Dataset: `sezenkarakus/image-description-dataset-v2` (HuggingFace) — 19.6k HD images (311–612px), diverse scenes with text descriptions.
- **D-02:** Storage layout: parquets at `D:\Images\train\parquets\`, extracted PNGs at `D:\Images\train\raw\`.
- **D-03:** `prepare_data.py` — standalone one-time script. Reads parquets from `D:\Images\train\parquets\`, saves each image as PNG + writes `descriptions.csv` (filename → text) to `D:\Images\train\raw\`. Text descriptions NOT used during training — CSV preserved for future conditioning experiments.
- **D-04:** `train.py` accepts `--data-dir` CLI arg (default: `D:\Images\train\raw`). DataLoader applies center-crop to square → resize to 64×64 on-the-fly (not baked into PNGs). Transforms use `torchvision.transforms`.

### Batched Environment

- **D-05:** Single `BatchedPaintEnv` class — holds all 96 canvases internally as `(96, 3, 64, 64)` tensors. All ops (`reset()`, `step()`, reward computation) operate on the full batch in one GPU call. No per-env Python loop. Matches ROADMAP "tensor ops batchées".
- **D-06:** All 96 envs reset together at `N_STROKES` steps (fixed episode length = 40 steps × 5 strokes = 200 total strokes). No asynchronous resets needed.
- **D-07:** `apply_strokes(canvas, actions, R)` — standalone function in `env.py` that applies k=5 strokes sequentially via R (stroke 1 composited first, then stroke 2 on top, etc.). Used by `env.step()` under `torch.no_grad()` and by `agent.update_step()` with gradients enabled (actor policy gradient flows through R → actor).

### Training Budget & Run Configuration

- **D-08:** `MAX_EPISODES` in `config.py` + `--max-episodes` CLI arg in `train.py`. Default: 10 000 episodes (baseline solide). Autoresearch uses 2000 for fast validation runs before the full run.
- **D-09:** `WARMUP_STEPS = 1000` in `config.py` — minimum transitions in the replay buffer before any gradient update begins. ~10 episodes × 96 envs. Agent plays with max exploration noise during warm-up.
- **D-10:** Checkpoints saved every 500 episodes to `checkpoints/` — `actor.pt`, `critic.pt`, and optimizer states. `train.py --resume-from checkpoints/ep_XXXX/` loads all states and resumes from that episode.
- **D-11:** TensorBoard logging every 100 steps: `Q_max`, mean episode reward, critic loss, mean action std (for noise annealing monitoring).

### Claude's Discretion

- Exact LR scheduler implementation (LinearLR vs manual step at `LR_DECAY_STEP=100_000`) — planner decides based on what's simplest.
- Reward edge case (L2_init ≈ 0 when canvas already matches target) — add epsilon guard: `reward = (L2_prev - L2_new) / max(L2_init, 1e-8)`.
- Whether `apply_strokes()` is a module-level function or a static method of `BatchedPaintEnv` — planner decides.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Paper (authoritative)
- `LearningToPaint.pdf` §3.2 — State representation: 7 channels (canvas 3 + target 3 + step 1). Confirmed in Phase 3 CONTEXT D-01.
- `LearningToPaint.pdf` §3.3.1 — Model-based DDPG: critic takes s_{t+1} (rendered next-state after k=5 strokes), not (s_t, a_t).
- `LearningToPaint.pdf` §3.3.2 — Action bundle: k=5, discount γ^k=0.955. Sequential application per bundle.
- `LearningToPaint.pdf` §3.4 — Actor BN, critic WN+TReLU. Actor policy gradient through frozen R.
- `LearningToPaint.pdf` Appendix §7.1, Table 1 — All hyperparameters (LRs, batch size, τ, γ). Already implemented in `config.py`.

### Existing Implementation (Phases 1–3)
- `models/renderer.py` — `SoftRasterizer`: `(batch, 8) → (batch, 3, 64, 64)`. Frozen (eval + requires_grad_(False)). Actor loss backpropagates through this.
- `pretrain_renderer.py::load_frozen_renderer` — canonical freeze-load pattern. Use in `env.py` to load R.
- `models/actor.py` — `Actor`: `(batch, 7, 64, 64) → (batch, 40)` via sigmoid. Output = k=5 strokes as flat vector.
- `models/critic.py` — `Critic`: `(batch, 7, 64, 64) → (batch, 1)` scalar V(s'). Takes rendered next-state.
- `ddpg/agent.py` — `DDPGAgent` with complete `update_step()` scaffold (see docstring for the intended Phase 4 implementation sequence — it's the authoritative spec for update_step).
- `ddpg/replay_buffer.py` — `ReplayBuffer`: 200k transitions, uint8 canvas, sample returns float32.
- `config.py` — ALL hyperparameters already defined: `ACTOR_LR=3e-4`, `CRITIC_LR=1e-3`, `ACTOR_LR_FINAL=1e-4`, `CRITIC_LR_FINAL=3e-4`, `LR_DECAY_STEP=100_000`, `GAMMA=0.955`, `TAU=0.005`, `BATCH_SIZE=96`, `REPLAY_BUFFER_CAPACITY=200_000`, `GRAD_CLIP_CRITIC=1.0`.

### Requirements
- `.planning/REQUIREMENTS.md` — TRAIN-01 through TRAIN-04: exact acceptance criteria for env, update_step, train.py, and noise annealing.
- `.planning/ROADMAP.md` §Phase 4 — 5 success criteria. Read all 5 before planning.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ddpg/agent.py::update_step` docstring — the Phase 3 scaffold contains the full intended update sequence as pseudocode comments. The planner should use this as the implementation spec for update_step, not re-derive from the paper.
- `pretrain_renderer.py::load_frozen_renderer` — use this to load R in `env.py`. Handles eval() + requires_grad_(False) correctly.
- `config.py` — All DDPG hyperparameters already there. `train.py` and `env.py` must import from config, not hardcode.
- `ddpg/replay_buffer.py::ReplayBuffer` — already has `push()` and `sample()`. `env.step()` results go into this buffer.

### Established Patterns
- Import convention: `from config import IMG_SIZE, N_STROKES, ...` (flat, no package-relative imports).
- Standalone scripts at project root: `prepare_data.py`, `train.py` follow the same pattern as `pretrain_renderer.py`.
- `ddpg/` is an existing sub-package — `ddpg/__init__.py` already exists. New modules drop in directly.
- Double-freeze pattern for frozen modules: `model.eval()` + `for p in model.parameters(): p.requires_grad_(False)`.

### Integration Points
- `env.py` imports: `SoftRasterizer` (from models.renderer), `config` constants, DataLoader for target images.
- `ddpg/agent.py::update_step` imports: `SoftRasterizer` (for actor loss differentiable compositing), `apply_strokes` (from env).
- `train.py` imports: `BatchedPaintEnv`, `DDPGAgent`, `ReplayBuffer`, `SoftWriter` (TensorBoard).
- New config constants needed: `MAX_EPISODES`, `WARMUP_STEPS` — add to `config.py` as part of Phase 4.

</code_context>

<specifics>
## Specific Ideas

- **Reward normalization edge case:** `reward = (L2_prev - L2_new) / max(L2_init, 1e-8)` to guard against zero-division when canvas already matches target at episode start.
- **apply_strokes() signature:** `apply_strokes(canvas: Tensor, actions: Tensor, R: SoftRasterizer) -> Tensor` — takes `(B, 3, 64, 64)` canvas + `(B, 40)` actions → returns `(B, 3, 64, 64)` new canvas after applying k=5 strokes sequentially. Same function called in env.step() (no_grad) and update_step() (with grad).
- **Checkpoint filename convention:** `checkpoints/ep_{episode:06d}/actor.pt`, `checkpoints/ep_{episode:06d}/critic.pt`, `checkpoints/ep_{episode:06d}/opt.pt` (optimizer states). `--resume-from checkpoints/ep_002000/` resumes from episode 2000.
- **Dataset loading:** `torchvision.datasets.ImageFolder` won't work for a flat folder with no class subdirs — use a custom `torch.utils.data.Dataset` that lists PNG files in the data-dir and applies transforms.
- **Autoresearch hook:** `MAX_EPISODES` in config.py means autoresearch can do quick 2k-episode validation runs by overriding just this constant, then run the full 10k at the end.

</specifics>

<deferred>
## Deferred Ideas

- **WGAN discriminator** — Evolution if agent produces "bouillie floue" after Phase 5 eval. Phase 6+ scope.
- **Text-conditioned painting** — The descriptions.csv sidecar is preserved for this future experiment. Not Phase 4 or 5 scope.
- **Best-checkpoint selection** — Saving every 500 episodes; selecting the best checkpoint for eval (by reward) is Phase 5 scope (eval.py picks the checkpoint).

</deferred>

---

*Phase: 4-Training Loop*
*Context gathered: 2026-06-10*
