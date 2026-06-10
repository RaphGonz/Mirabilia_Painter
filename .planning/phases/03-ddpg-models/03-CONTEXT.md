# Phase 3: DDPG Models - Context

**Gathered:** 2026-06-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement the four DDPG components — actor, model-based V(s') critic, target networks, and replay buffer — with correct architectures and shapes, ready to be wired into the training loop in Phase 4.

Deliverables:
- `models/actor.py` — ResNet18 + CoordConv, `(batch, 7, 64, 64) → (batch, 40)` via sigmoid
- `models/critic.py` — ResNet18 + CoordConv + WN+TReLU, `(batch, 7, 64, 64) → (batch, 1)` scalar V(s')
- `ddpg/agent.py` — target network management, soft update, actor/critic update step
- `ddpg/replay_buffer.py` — numpy ring buffer, 200k transitions, uint8 canvas storage

Out of scope: env.py, training loop, compositing logic, exploration noise annealing (Phase 4).

</domain>

<decisions>
## Implementation Decisions

### State representation (CORRECTION to paint_ai_design.md and ROADMAP)

- **D-01:** State is **7 channels** for BOTH actor and critic: `canvas (3) + target (3) + step_normalized (1)`. This is `s_t = (C_t, I, t/N_STROKES)` per §3.2 of the paper. `paint_ai_design.md` incorrectly documents 6 channels — disregard that file on this point.
- **D-02:** Step number encoded as scalar `t/N_STROKES ∈ [0, 1]`, broadcast (tiled) to a full `(1, 64, 64)` spatial channel before concatenation.
- **D-03:** Critic receives `s_{t+1}` (next-state after rendering the actor's proposed strokes), not `(s_t, a_t)`. Per §3.3.1: "the critic takes s_{t+1} as input rather than both s_t and a_t." Critic input shape: `(batch, 7, 64, 64)` — ROADMAP success criterion DDPG-02 had this wrong as 6ch; 7ch is correct.

### Actor architecture

- **D-04:** Follow paper Fig. 13 exactly, adapted for 64×64 input:
  - First layer: **3×3 CoordConv**, stride 2 → `(batch, 64, 32, 32)`. CoordConv appends normalized (x, y) coordinate channels before the convolution so the network has explicit spatial position information.
  - Backbone: **ResNet18-like** (4 residual stages: 2+2+2+2 blocks) + **BatchNorm** → global average pool → `(batch, 512)`
  - Head: `FC(512, 40)` + `sigmoid` → `(batch, 40)` in `[0, 1]`
- **D-05:** Output `40 = k × STROKE_DIM = 5 × 8`. Sigmoid (not tanh+rescale) for clean [0,1] output.
- **D-06:** BatchNorm in actor — confirmed by paper §3.4 ("The actor works well with Batch Normalization"). Actor is never called single-sample at inference (env.step uses batch=1 during rollout, but actor eval mode handles this fine with BN since we use running stats).

### Critic architecture

- **D-07:** Follow paper Fig. 14 exactly, adapted for 64×64 input:
  - First layer: **3×3 CoordConv**, stride 2 → `(batch, 64, 32, 32)`
  - Backbone: **ResNet18-like** (same structure as actor) + **Weight Normalization + TReLU** (NOT BatchNorm) → global average pool → `(batch, 512)`
  - Head: `FC(512, 1)` → scalar V(s'), no activation
- **D-08:** WN+TReLU on critic, not BatchNorm. Paper explicitly avoids BN on critic (§3.4: "BN can not speed up the critic learning significantly"). WN is applied at init via `torch.nn.utils.weight_norm()` wrapper.
- **D-09:** TReLU (Translated ReLU) from [Xiang & Li 2017] — shifts the activation threshold. Simplest implementation: `TReLU(x) = ReLU(x + bias)` where bias is a learned scalar per channel. Can use `nn.PReLU` as a practical approximation, or implement directly.

### Target networks

- **D-10:** Target actor and target critic = `copy.deepcopy()` at init, permanently in `eval()` mode.
- **D-11:** Soft update `τ=0.005`: `θ_target ← τ·θ + (1-τ)·θ_target`, applied after every critic gradient step.
- **D-12:** Target networks used in the Bellman target: `y = r + γ·V_target(s_{t+1})`. Target actor used only to generate `s_{t+1}` for the critic target (renders strokes from target actor's action, then critic_target evaluates the result).

### Replay buffer

- **D-13:** **Numpy ring buffer**, pre-allocated at init. 5 arrays: `obs_buf`, `act_buf`, `rew_buf`, `next_obs_buf`, `done_buf`. O(1) insert and sample.
- **D-14:** Capacity: **200k transitions** (deliberate deviation from paper's 800 episodes). Rationale: training on MS COCO-like diverse images requires larger buffer for sufficient sample diversity and reduced temporal correlation. At 40 steps/episode, 200k ≈ 5000 episodes.
- **D-15:** Canvas channels stored as **uint8** [0, 255] to minimize memory. Step channel stored as float32 (scalar, cheap). At 200k transitions, `obs_buf` ≈ 200k × 7 × 64 × 64 ≈ 5.7 GB (uint8 for 6 spatial channels, float32 for step channel). Implementation: store canvas (6ch) as uint8 array + step values as float32 array separately for memory efficiency.
- **D-16:** Target image stored redundantly per transition (no deduplication). Simpler sampling, no episode-index lookup logic.
- **D-17:** `sample(batch_size)` returns float32 tensors with canvas channels normalized to `[0, 1]` (divide uint8 by 255). Step channel passed through as-is.

### Hyperparameters (from paper Table 1)

- **D-18:** Actor LR: `3e-4`, decays to `1e-4` after 1e5 training batches. Critic LR: `1e-3`, decays to `3e-4`. Use `torch.optim.lr_scheduler.LinearLR` or manual step.
- **D-19:** Discount factor: `γ^k = 0.955` where k=5 (action bundle). This is the effective per-step discount, not per-stroke. Phase 4 will use this directly.
- **D-20:** Batch size: `96` (from paper). Phase 4 sets this.
- **D-21:** Gradient clipping on critic: `max_norm=1.0` (already in ROADMAP). No clipping on actor.

### File structure

- **D-22:** `ddpg/` directory for `agent.py` and `replay_buffer.py` (as in `paint_ai_design.md` file tree). `models/actor.py` and `models/critic.py` alongside existing `models/renderer.py`.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Paper (authoritative — overrides all project docs on architecture)
- `LearningToPaint.pdf` §3.2 — State representation: s_t = (C_t, I, t). 7 channels confirmed.
- `LearningToPaint.pdf` §3.3.1 — Model-based DDPG: critic takes s_{t+1}, V(s) formulation.
- `LearningToPaint.pdf` §3.3.2 — Action Bundle: k=5, discount adjustment γ^k=0.955.
- `LearningToPaint.pdf` §3.4 — Network architectures: actor BN, critic WN+TReLU, CoordConv.
- `LearningToPaint.pdf` Appendix §7.1, Fig. 13 & 14, Table 1 — Exact architecture diagrams and hyperparameters. READ THESE before coding.

### Existing Implementation
- `models/renderer.py` — `SoftRasterizer` (and `NeuralRenderer` alias): `(batch, 8) → (batch, 3, 64, 64)`. Frozen in Phase 4. actor loss backpropagates through this.
- `config.py` — `IMG_SIZE=64`, `STROKE_DIM=8`, `STROKES_PER_STEP=5`, `N_STROKES=40`, `RENDERER_BETA`.
- `renderer.py` — Hard rasterizer `draw()`. Not used by DDPG models directly, but reference for compositing logic.
- `models/__init__.py` — Already exists (empty). Add actor.py and critic.py alongside renderer.py.

### Design
- `paint_ai_design.md` — File structure, design decisions, evolution plan. **NOTE: state channel count (6ch) in this file is WRONG — it's 7ch per paper. All other content valid.**
- `.planning/REQUIREMENTS.md` — DDPG-01 through DDPG-04 acceptance criteria.
- `.planning/ROADMAP.md` §Phase 3 — Success criteria. **NOTE: DDPG-02 says critic input is (batch, 6, 64, 64) — WRONG. Correct is (batch, 7, 64, 64) per paper Fig. 14.**

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `models/renderer.py::SoftRasterizer` — import as `from models.renderer import SoftRasterizer`. Actor loss in Phase 4 will call `R(actor_output)` to get `(batch, 3, 64, 64)` stroke images for compositing into next-state.
- `config.py::IMG_SIZE, STROKE_DIM, STROKES_PER_STEP, N_STROKES` — import directly; no magic numbers in actor.py, critic.py, or replay_buffer.py.
- `pretrain_renderer.py::load_frozen_renderer` — canonical freeze-load pattern for Phase 4 env.py. Reference it when documenting how Phase 4 should load R.

### Established Patterns
- Import convention: `from config import IMG_SIZE` (not package-relative).
- No BatchNorm in SoftRasterizer; actor uses BN, critic uses WN+TReLU — explicit divergence is intentional and paper-backed.
- `models/__init__.py` already exists (empty) — new modules drop in directly.
- Flat root for standalone scripts (`pretrain_renderer.py`). `ddpg/` is a new sub-package; needs `ddpg/__init__.py`.

### Integration Points
- `models/actor.py` and `models/critic.py` will be imported by `ddpg/agent.py`.
- `ddpg/replay_buffer.py` is standalone (no dependency on models).
- `ddpg/agent.py` imports actor, critic, and SoftRasterizer (for actor loss computation — Phase 4).
- Phase 4 `env.py` will call `actor.forward(obs)` and `R.forward(action)` at every step.

</code_context>

<specifics>
## Specific Ideas

- **CoordConv implementation:** Append two channels to the input — one with x-coordinates normalized to [-1, 1], one with y-coordinates normalized to [-1, 1]. Then apply standard Conv2d on the (7+2=9)-channel input. Total input channels to CoordConv: 9.
- **ResNet18 adaptation for 64×64:** Standard ResNet18 has a 7×7 conv with stride 2 as first layer (designed for 224×224). Since our CoordConv already does stride-2 to 32×32, the ResNet18 backbone should start from stage 1 (3×3 convs, no aggressive downsampling) rather than the original stem. Use a simplified ResNet18: 4 stages of [2, 2, 2, 2] BasicBlocks, channels [64→64→128→256→512], starting from 32×32 spatial input.
- **TReLU:** Simplest implementation is `F.relu(x + self.bias)` where `self.bias` is `nn.Parameter(torch.zeros(C))` initialized to zero. Applied per-channel after WN convolutions in the critic.
- **Buffer memory estimate:** 200k transitions × (6 channels uint8 + 1 channel float32) × 64×64 per obs = ~(200k × 6 × 64 × 64 + 200k × 1 × 64 × 64 × 4) bytes ≈ 4.9 GB + 3.3 GB = ~8 GB for obs+next_obs combined. May need to store step channel as scalar (not tiled) to reduce this. Planner should benchmark memory before deciding to tile or compute step channel on-the-fly during sampling.

</specifics>

<deferred>
## Deferred Ideas

- **WGAN discriminator** — Phase 4+ evolution when "bouillie floue" symptom appears. `models/discriminator.py`, PatchGAN-like architecture (Fig. 11 in paper appendix).
- **LR scheduler** — actor/critic LR decay after 1e5 batches. Captured in D-18 for Phase 4 implementation.
- **CoordConv with radius channel** — some implementations add a 3rd channel with distance from center. Paper doesn't use it; skip for baseline.

</deferred>

---

*Phase: 3-DDPG Models*
*Context gathered: 2026-06-10*
