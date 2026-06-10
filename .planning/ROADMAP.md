# Roadmap: Paint AI — Mirabilia Episode 1

## Overview

Build a DDPG-based painting agent from scratch in PyTorch that learns to reproduce target images by laying sequential opaque rectangular strokes on a 64x64 canvas. The build follows a strict dependency order: hard rasterizer first, then a differentiable neural renderer R (pre-trained and frozen — hard gate before any RL), then the DDPG model architecture, then the full training loop, and finally an eval pipeline that produces a viewable timelapse. Every phase delivers a verifiable capability; Phase 2 is a blocking gate that nothing downstream can bypass.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation** - Config constants, palette projection, and the hard rasterizer
- [ ] **Phase 2: Neural Renderer** - Pre-train R, visual validation hard gate, freeze verification
- [ ] **Phase 3: DDPG Models** - Actor CNN, model-based V(s') critic, target networks, replay buffer
- [ ] **Phase 4: Training Loop** - env.py, agent update loop, train.py with 96 parallel envs, exploration noise
- [ ] **Phase 5: Eval & Timelapse** - Deterministic rollout, palette projection, timelapse GIF/MP4 export

## Phase Details

### Phase 1: Foundation

**Goal**: All shared infrastructure exists — constants, palette, and hard rasterizer — so every downstream module can import from a stable base.
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: FOUND-01, FOUND-02, FOUND-03
**Success Criteria** (what must be TRUE):

  1. `config.py` can be imported from any module and exposes `IMG_SIZE=64`, `STROKE_DIM=8`, `STROKES_PER_STEP=5`, `N_STROKES`, and `IMAGE_RANGE=(0.0, 1.0)` with the correct types and values
  2. `palette.py` accepts a manual RGB list and `project_color(rgb, colorspace)` returns the nearest palette color for all three colorspaces (`rgb`, `oklab`, `hsv`) without error
  3. `renderer.py` `draw(canvas, stroke_params)` renders a visually correct opaque oriented rectangle on the canvas for arbitrary `(cx, cy, w, h, θ, r, g, b)` params including edge cases (thin strokes, full-canvas strokes, extreme rotations)
  4. All three modules are importable with no circular dependencies and no autograd graph attached to the hard rasterizer output

**Plans**: 2 plans

- [x] 01-01-PLAN.md — Scaffold + config.py constants + palette.py projection + Wave 0 test scaffold (FOUND-01, FOUND-02)
- [x] 01-02-PLAN.md — renderer.py hard rasterizer (oriented rectangle, pure PyTorch, no autograd) + visual gate (FOUND-03)

### Phase 2: Neural Renderer

**Goal**: Differentiable renderer R is implemented, visually validated on out-of-distribution strokes, frozen, and verified to stay frozen — acting as a hard gate before any RL work begins.
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: REND-01, REND-02, REND-03

> **ARCHITECTURAL PIVOT (autoresearch série 1–4, 2026-06-10):** The `NeuralRenderer` CNN (trained, `renderer.pkl`-based) was **replaced by `SoftRasterizer`** — an analytical differentiable soft rasterizer using a sigmoid SDF approximation. No supervised pretraining is required. `renderer.pkl` is obsolete. The `NeuralRenderer` name is kept as a backward-compat alias.
>
> Formula: `alpha(x,y) = sigmoid((w/2 - |dx’|) / β) * sigmoid((h/2 - |dy’|) / β)` where β=1.0 (~4px edge softness). Output: `(B, 3, H, W)` premultiplied `(alpha * color)`. Compositing: `new_canvas = alpha * color + (1 - alpha) * old_canvas`.

**Success Criteria** (what must be TRUE):

  1. ✅ R (`models/renderer.py::SoftRasterizer`) accepts input `(batch, 8)` and produces output `(batch, 3, 64, 64)` in `[0, 1]` with no BatchNorm layers
  2. ✅ N/A — no supervised pretraining required (SoftRasterizer is analytical). `pretrain_renderer.py` sampling helpers retained for Phase 3 env.py use.
  3. ✅ Visual inspection via `visual_gate.png` confirms recognizable, non-smeared soft rectangles across 8 stroke types (thin, tilted, edge, full-canvas, extreme theta)
  4. ✅ After freeze, param norm unchanged after forward pass — SoftRasterizer has no learned params; `.eval()` + `requires_grad_(False)` verified in `tests/test_neural_renderer.py`
  5. ✅ HARD GATE CLEARED — Phase 3 unblocked

**Plans**: 2 plans (complete)

- [x] 02-01-PLAN.md — Initial NeuralRenderer CNN scaffold (superseded by autoresearch pivot to SoftRasterizer)
- [x] 02-02-PLAN.md — pretrain_renderer.py scaffold + autoresearch → SoftRasterizer pivot (REND-01, REND-02, REND-03)

### Phase 3: DDPG Models

**Goal**: All four DDPG components — actor, model-based V(s') critic, target networks, and replay buffer — are implemented with correct shapes and architecture, ready for the training loop.
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: DDPG-01, DDPG-02, DDPG-03, DDPG-04
**Success Criteria** (what must be TRUE):

  1. Actor (`models/actor.py`) accepts `(batch, 7, 64, 64)` and produces `(batch, 40)` output in `[0, 1]` via sigmoid; a shape assertion test passes for a single forward pass
  2. Critic (`models/critic.py`) accepts the rendered next-state image `(batch, 7, 64, 64)` and produces a scalar Q value — it does NOT accept raw action floats; a shape assertion confirms the input is a 7-channel image tensor, not a concatenation of state and action vectors. (NOTE: 7ch per CONTEXT.md D-03 — earlier "6ch" wording was incorrect.)
  3. Target networks are deepcopies of actor and critic, permanently in `eval()` mode; soft update with `τ=0.005` produces target params that are a weighted average of current and previous target params after a single update call
  4. Replay buffer stores 200k transitions with canvas tensors in `uint8`, and sampling returns `float32` tensors with correct shapes for all five fields (obs, action, reward, next_obs, done)

**Plans**: 4 plans

- [x] 03-01-PLAN.md — models/actor.py (ResNet18+CoordConv+BN → (B,40) sigmoid) + tests (DDPG-01)
- [x] 03-02-PLAN.md — ddpg/replay_buffer.py (200k numpy ring buffer, uint8 canvas, scalar step) + tests (DDPG-04)
- [ ] 03-03-PLAN.md — models/critic.py (ResNet18+CoordConv+WN+TReLU → (B,1) V(s')) + tests (DDPG-02)
- [ ] 03-04-PLAN.md — ddpg/agent.py (deepcopy target nets, soft update τ=0.005, update_step scaffold) + tests (DDPG-03)

### Phase 4: Training Loop

**Goal**: The full DDPG training loop runs end-to-end — environment, agent updates, 96 parallel envs, exploration noise annealing — and shows measurable reward improvement over the first 1000 episodes.
**Mode:** mvp
**Depends on**: Phase 3
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04
**Success Criteria** (what must be TRUE):

  1. `env.py` `reset()` returns a `(7, 64, 64)` observation and `step(action)` applies k=5 strokes sequentially via R (not in parallel against the same base canvas), returns normalized L2 incremental reward `(L2_prev - L2_new) / L2_init`, and terminates correctly at N_STROKES
  2. Agent update applies MSE critic loss, policy gradient through frozen R (`torch.no_grad()` on R), soft-updates targets, and clips critic gradients at `max_norm=1.0`; a single update call on dummy data completes without error and does not modify R parameters
  3. `train.py` runs with 96 batched envs, logs `Q_max`, mean reward, and critic loss to TensorBoard every 100 steps, and is launchable from CLI with configurable arguments
  4. Mean episode reward shows an upward trend over the first 1000 episodes (raw TensorBoard curve slopes positive); `Q_max` stays below 1.0 throughout, confirming no Q-value explosion under normalized L2 reward
  5. Exploration noise anneals from σ=0.3 to σ=0.05 over the training run; per-episode mean action std is logged and visibly decreasing in TensorBoard

**Plans**: TBD

### Phase 5: Eval & Timelapse

**Goal**: The trained agent paints a target image in a deterministic rollout, colors are projected to the palette, the final canvas is rendered via the hard rasterizer, and a viewable timelapse GIF/MP4 is exported — the central deliverable of Mirabilia Episode 1.
**Mode:** mvp
**Depends on**: Phase 4
**Requirements**: EVAL-01, EVAL-02
**Success Criteria** (what must be TRUE):

  1. `eval.py` runs a deterministic rollout (no exploration noise), projects all RGB stroke colors to the nearest palette color via nearest-neighbor, then replays the ordered stroke list through the hard rasterizer to produce a final clean canvas that is visually closer to the target than a blank canvas
  2. `eval.py` exports a timelapse as both GIF and MP4 (via imageio) showing the agent painting the target image stroke by stroke; the exported file is playable and the target image is recognizable in the final frame

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 2/2 | Complete | 2026-06-09 |
| 2. Neural Renderer | 2/2 | Complete | 2026-06-10 |
| 3. DDPG Models | 2/4 | In progress | - |
| 4. Training Loop | 0/TBD | Not started | - |
| 5. Eval & Timelapse | 0/TBD | Not started | - |
