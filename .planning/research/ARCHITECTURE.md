# Architecture Research — Paint AI

**Domain:** RL-based computational painting (stroke-based image synthesis)
**Reference:** "Learning to Paint With Model-Based Deep Reinforcement Learning" (Huang et al., ICCV 2019)
**Researched:** 2026-06-08
**Confidence:** HIGH — cross-verified against original paper (ar5iv HTML), reference GitHub implementation (hzwer/ICCV2019-LearningToPaint), and design document

---

## System Components

### 1. Hard Rasterizer (`renderer.py`)

**What it does:** Renders a single oriented rectangle onto a canvas tensor using deterministic pixel math. Pure tensor operations; no learned weights; no gradient required.

**Boundary:** Takes `(canvas, stroke_params)` where `params = (cx, cy, w, h, θ, r, g, b)` (STROKE_DIM=8). Returns updated canvas tensor. Does NOT know about RL, training, or the neural renderer.

**When it is active:**
- During neural renderer pre-training: generates supervised training pairs `(params → ground-truth stroke image)`
- At inference / eval time: final timelapse rendering, replaying the ordered stroke list for a crisp result
- In reward computation if the reward function operates on a hard-rendered canvas rather than the soft one (project choice: reward runs on soft canvas — see `env.py` discussion)

**Constraints:** No alpha blending — rectangles are opaque, overwrite pixels. Composition order matters.

---

### 2. Neural Renderer R (`models/renderer.py`)

**What it does:** A learned neural network that approximates the hard rasterizer. Takes stroke parameters → outputs the stroke image in isolation (not composited onto a canvas). Trained once via supervised learning, then frozen.

**Architecture (per paper):** FC layers + convolutional layers + sub-pixel upsampling. Input: flat vector of stroke params (8 floats). Output: single stroke image at canvas resolution (3×H×W or with an alpha mask channel).

**Key design choice — renders the stroke alone, not the composited canvas.** Composition is done outside the network by the environment's `decode()` function. This preserves explicit occlusion control.

**Boundary:**
- Input: stroke params tensor (batch of STROKE_DIM floats per stroke)
- Output: stroke image (H×W×3) + optional binary mask
- Has no knowledge of the current canvas, the target image, or the RL agent
- Parameters are frozen after pre-training; no gradient flows back into R's weights during RL training

**Why it exists:** Differentiability. The hard rasterizer cannot be differentiated with respect to stroke positions. R can. Gradient from the actor loss flows through R's forward pass (not back into R's weights) to improve the actor.

---

### 3. RL Environment (`env.py`)

**What it does:** Mediates between the DDPG agent and the canvas. Implements `reset()` and `step(action) → (obs, reward, done, info)`. Owns the canvas tensor and step counter.

**State construction:** `obs = concat(canvas_current, target_image)` → shape `(6, H, W)` for RGB. The reference implementation adds a 7th channel: a normalized timestep scalar broadcast to `(1, H, W)`, giving `(7, H, W)` total. This is worth replicating — the step count lets the actor know how much of the episode budget remains.

**Step logic:**
1. Unpack action into k=5 stroke param vectors
2. For each stroke, call `R(params)` to get the stroke image
3. Composite each stroke onto the current canvas (simple overwrite / alpha blend)
4. Compute incremental L2 reward: `r = L2(canvas_prev, target) − L2(canvas_new, target)`
5. Normalize reward by initial L2 distance (prevents scale issues across images)
6. Increment step counter; set `done = (step >= N_STROKES / k)`
7. Return new observation

**Batched environments:** The reference implementation runs 96 environments in parallel (`env_batch=96`). This is not a sequential loop — it is a batched tensor operation. The environment should be designed for batch-first tensors from the start.

**Boundary:** Owns canvas state. Calls R for forward rendering. Calls reward module. Returns (obs, r, done). Does not own the DDPG agent or training logic.

---

### 4. Reward Module (`reward.py`)

**What it does:** Computes the scalar reward signal.

**Formula:** `r_t = (L2(canvas_{t-1}, target) − L2(canvas_t, target)) / (L2(canvas_0, target) + ε)`

- Numerator: improvement in L2 distance this step (positive = got closer to target)
- Denominator: initial L2 at episode start — normalizes reward scale across images
- Result: dimensionless fraction in roughly [−1, 1] for well-behaved episodes

**Boundary:** Stateless function. Takes `(canvas_prev, canvas_new, target, initial_l2)` → scalar reward. No network weights.

**Decoupling note:** Separating reward into its own module (`reward.py`) is architecturally important. The WGAN reward (episode 2 evolution) replaces only this module; everything else stays the same.

---

### 5. Actor Network (`models/actor.py`)

**What it does:** The policy. Given the current observation, outputs a bundle of k=5 stroke parameter vectors simultaneously.

**Architecture:** CNN encoder (ResNet-style) on the (7, H, W) input, followed by FC layers, final tanh activation → output in [0, 1]^(STROKE_DIM × k) = [0, 1]^40.

**Key behavioral note:** All k strokes are decided from the same state snapshot. The actor does NOT see the canvas updated by strokes 1..j when deciding stroke j+1 within the same bundle. This is the known intra-bundle blind spot — accepted in the baseline.

**Coordinate convolution:** The reference implementation adds a CoordConv channel (x/y position grids) to help the actor reason about spatial positions. Worth including.

**Boundary:**
- Input: observation tensor (batch, 7, H, W)
- Output: action tensor (batch, STROKE_DIM × k) clamped to [0, 1]
- No direct connection to canvas, renderer, or reward
- Receives gradients from critic during DDPG update

---

### 6. Critic Network (`models/critic.py`)

**What it does:** Estimates Q(state, action) — the expected cumulative discounted reward from state `s` taking action `a` and then following the policy.

**Architecture:** Same ResNet backbone as actor, but takes `(state, action)` as input. The action vector is injected at a late FC layer (not at the image-level input). Uses weight normalization (no batch norm) for stability with off-policy samples.

**In the model-based variant (paper):** The critic takes `s_{t+1}` (the next state obtained by running the renderer on the action) rather than raw `(s_t, a_t)`. The transition function `s_{t+1} = decode(s_t, a_t)` is differentiable, so gradients can flow from the critic through the rendered next state back to the actor. This is the "model-based" leverage: the actor is trained by gradient through the renderer, not just through the critic's Q estimate.

**Boundary:**
- Input: (state, action) or (next_state) depending on model-based vs. model-free variant
- Output: scalar Q-value
- No knowledge of the replay buffer or environment; receives batches from the agent

---

### 7. DDPG Agent (`ddpg/agent.py`)

**What it does:** Owns the training logic. Holds actor, critic, their target network copies, and the replay buffer. Implements `select_action()`, `store_transition()`, `update_policy()`.

**Target networks:** Both actor and critic have frozen-update copies (`actor_target`, `critic_target`). Initialized to match the main networks. Updated via soft update: `θ_target ← τ·θ + (1−τ)·θ_target`, with τ=0.001.

**update_policy() steps:**
1. Sample minibatch from replay buffer
2. Compute target Q: `y = r + γ · Q_target(s', actor_target(s'))`
3. Update critic: minimize MSE between `Q(s, a)` and `y`
4. Update actor: maximize `Q(s, actor(s))` (policy gradient)
5. In model-based variant: also maximize `r(s, actor(s))` computed via R (direct gradient through renderer)
6. Soft-update both target networks

**Boundary:** Orchestrates training. Calls actor, critic, target networks, replay buffer. Does NOT call env directly — transitions come in via `store_transition()` from the training loop.

---

### 8. Replay Buffer (`ddpg/replay_buffer.py`)

**What it does:** Circular buffer of transitions `(s, a, r, s', done)`. Enables off-policy training by decorrelating consecutive samples.

**Capacity:** Reference uses 800 transitions (small by RL standards — justified by episode-length normalization and image-space states being large). For 96 parallel envs, this fills quickly.

**Boundary:** Passive storage. Accepts `push(transition)`, returns `sample(batch_size)`. No network weights, no training logic.

---

### 9. Pre-training Script (`pretrain_renderer.py`)

**What it does:** Generates random stroke parameters, renders them with the hard rasterizer to get ground-truth images, and trains R to match. Saves `renderer.pkl` when converged.

**Boundary:** Only component allowed to call both hard rasterizer and R simultaneously. After this script completes, the hard rasterizer and R are only called separately by different pipeline stages.

---

### 10. Training Loop (`train.py`)

**What it does:** Outer orchestrator. Runs episodes, feeds transitions to the agent, triggers `update_policy()`, logs metrics, saves checkpoints.

**Boundary:** Calls `env.step()`, `agent.select_action()`, `agent.store_transition()`, `agent.update_policy()`. Does not own any network weights directly.

---

### 11. Eval / Inference (`eval.py`)

**What it does:** Loads a trained actor. Given a target image, runs the agent step-by-step, collects stroke param sequences, applies palette nearest-neighbor projection, and renders the final sequence using the hard rasterizer for the crisp output. Optionally exports frames for timelapse.

**Boundary:** Calls actor (inference only, no gradients). Calls hard rasterizer. Calls palette module. Reads `actor.pkl`. Does not touch R at inference.

---

### 12. Config (`config.py`)

Single source of truth for all hyperparameters: `STROKE_DIM=8`, `STROKES_PER_STEP=5`, `IMG_SIZE=64`, `N_STROKES`, `BATCH_SIZE`, `REPLAY_SIZE`, `TAU`, learning rates. Every other module imports from here.

---

### 13. Palette Module (`palette.py`)

Holds the ~40 palette colors. Exposes `project(rgb_continuous) → palette_color` via nearest-neighbor L2. Used only at inference time by `eval.py`.

---

## Data Flow

### Pre-Training Phase (one-time, blocking)

```
config.py
    │
    ▼
[Random stroke params sampled uniformly from [0,1]^8]
    │
    ▼
Hard Rasterizer ──────────────────────► ground-truth stroke image
    │                                         │
    │                                         ▼
    └──► Neural Renderer R (forward pass) ──► predicted stroke image
                │
                ▼
         MSE loss (predicted vs ground-truth)
                │
                ▼
         Backprop → update R weights
                │
                ▼
         [repeat until convergence]
                │
                ▼
         Save renderer.pkl → R is now FROZEN
```

### RL Training Phase (main training)

```
Target image (from dataset)
    │
    ▼
env.reset()
    ├── canvas ← blank (zeros)
    ├── step_count ← 0
    └── obs ← concat(canvas, target, step_channel)  [shape: batch×7×H×W]
                                    │
                                    ▼
                           Actor(obs) → action [batch×40]
                                    │
                           (+ exploration noise during training)
                                    │
                                    ▼
                           env.step(action)
                                    │
                            ┌───────┴────────┐
                            │                │
                     For each of k=5 strokes:│
                     R(stroke_params)         │
                     → stroke image           │
                     composite onto canvas    │
                            │                │
                            └───────┬────────┘
                                    │
                        new_canvas (via neural renderer R)
                                    │
                         ┌──────────┴──────────────┐
                         │                          │
                  reward module              increment step
                  r = ΔL2 / L2_init         new obs constructed
                         │
                         ▼
              Replay Buffer ← push(obs, action, reward, new_obs, done)
                         │
                         ▼ (after warmup, once per episode end)
              Agent.update_policy()
                    │
              ┌─────┼─────┐
              │           │
         Critic      Actor
         update      update
         (MSE Q)     (policy gradient
                      + model-based
                      gradient through R)
              │           │
              └─────┬─────┘
                    │
             Soft-update target networks
                    │
             (loop continues)
```

### Inference Phase

```
Trained actor.pkl + Target image
    │
    ▼
eval.py: run agent step by step
    │
    ▼
Actor(obs) → action [deterministic, no noise]
    │
    ▼
Apply bundle to internal canvas
    │
    ├── Collect stroke params list (ordered)
    ├── Apply stop threshold (gain L2/step < ε)
    │
    ▼
Palette projection: RGB_continuous → nearest palette color
    │
    ▼
Replay ordered stroke list through Hard Rasterizer
    │
    ▼
Final crisp canvas + frame sequence for timelapse export
```

---

## Training Pipeline

### Phase 1 — Neural Renderer Pre-training

**Goal:** R learns to approximate the hard rasterizer well enough that gradients through R are informative.

**Process:**
- Sample random stroke params uniformly across the full parameter space (critical: cover corners — tiny strokes, edge-touching strokes, all rotations)
- Render each with the hard rasterizer → ground-truth image
- Feed same params to R → predicted image
- Loss: per-pixel MSE (L1 also viable, slightly sharper results)
- Train until validation MSE plateaus
- Checkpoint as `renderer.pkl`

**Validation criterion:** Visual inspection of R output vs hard rasterizer for held-out strokes. Pay attention to thin/rotated strokes — these are the failure mode.

**Duration:** Typically hours on a GPU. Does not require a dataset of real images — purely synthetic.

**Hard gate:** RL training MUST NOT begin until this phase is complete and validated. A poor R produces uninformative gradients that mask RL bugs.

---

### Phase 2 — DDPG RL Training

**Goal:** Actor learns a policy that maps (canvas, target) → stroke bundle to minimize L2 over the episode.

**Process:**
- Initialize canvas to blank, sample target image from dataset
- Collect transitions with the actor + exploration noise
- Fill replay buffer during warmup (no gradient updates yet)
- After warmup: run `update_policy()` 10× per episode
- Decay learning rate and exploration noise over training
- Checkpoint actor every N episodes

**Parallel environments:** Run 96 envs simultaneously. The env's `step()` must be vectorized (batched tensor ops on GPU, not Python loops).

**Training signal:** Incremental L2 reward, normalized by initial L2. This keeps rewards in ~[−1, 1] regardless of image difficulty.

---

## Inference Pipeline

1. Load `actor.pkl`, load target image
2. `env.reset()` → blank canvas, obs constructed
3. Repeat until stop condition:
   a. `actor(obs)` → action (no noise, no gradient)
   b. Internally composite strokes via R (or hard rasterizer — either works for eval, but R is faster and already loaded)
   c. Check stop: if `L2(canvas, target) / L2_prev < 1 − ε` threshold not met for T consecutive steps → stop
4. Collect full ordered list of (params, palette-projected color)
5. Replay through hard rasterizer stroke by stroke → clean output
6. Save frames at each step → timelapse

**Train/inference gap:** The canvas seen during training is soft (neural renderer). The final output uses the hard rasterizer. Strokes whose quality relied on neural blending at boundaries may look slightly different. This is acceptable when strokes are large relative to the blending band.

---

## Build Order

Dependencies flow strictly downward. Nothing in a later tier can be tested until the tier it depends on exists and passes its own validation.

### Tier 1 — Zero dependencies (build first, in any order)

| Component | File | Validation |
|-----------|------|------------|
| Config | `config.py` | Import check; all constants accessible |
| Palette | `palette.py` | `project([0.5, 0.2, 0.8])` returns a valid palette color |
| Hard Rasterizer | `renderer.py` | Visual: `draw(blank_canvas, params)` → correct rectangle appears |

### Tier 2 — Depends on Tier 1

| Component | File | Depends on | Validation |
|-----------|------|-----------|------------|
| Neural Renderer R | `models/renderer.py` | config | Forward pass: `R(random_params)` → tensor of correct shape |
| Reward Module | `reward.py` | config | Unit test: reward is positive when canvas improves, negative when worse |
| Replay Buffer | `ddpg/replay_buffer.py` | config | Push N transitions, sample batch: shapes correct |

### Tier 3 — Depends on Tier 1+2

| Component | File | Depends on | Validation |
|-----------|------|-----------|------------|
| Pre-training script | `pretrain_renderer.py` | Hard rasterizer, R | R loss decreases; visual: R output matches hard rasterizer on held-out params |
| Actor | `models/actor.py` | config | Forward pass: `actor(obs)` → shape (batch, 40), values in [0,1] |
| Critic | `models/critic.py` | config | Forward pass: `critic(obs, action)` → shape (batch, 1) |

**Hard gate between Tier 3 parts:** Pre-training must complete and be validated before Actor/Critic are wired to R. You can implement Actor and Critic in parallel with pre-training, but integration testing waits.

### Tier 4 — Depends on Tier 3

| Component | File | Depends on | Validation |
|-----------|------|-----------|------------|
| Environment | `env.py` | Hard rasterizer, R (frozen), reward, config | `env.step(random_action)` returns obs of correct shape, reward is scalar, done is bool |
| DDPG Agent | `ddpg/agent.py` | Actor, critic, replay buffer, config | `agent.update_policy()` runs without error on a batch of dummy transitions; losses are finite |

### Tier 5 — Depends on Tier 4

| Component | File | Depends on | Validation |
|-----------|------|-----------|------------|
| Training loop | `train.py` | Env, DDPG agent | Reward increases over first 1000 episodes; no NaN/Inf losses |
| Eval / Timelapse | `eval.py` | Actor (trained), hard rasterizer, palette, env | Output image visually resembles target; frames exportable as GIF |

---

## Integration Points

These are the explicit interface contracts between components. Violating these shapes causes silent failures that are hard to trace.

### R — Neural Renderer Interface

```
Input:  params  : Tensor[batch, STROKE_DIM]       # floats in [0, 1]
Output: stroke  : Tensor[batch, 3, IMG_SIZE, IMG_SIZE]  # float in [0, 1]
        mask    : Tensor[batch, 1, IMG_SIZE, IMG_SIZE]  # binary or soft [0, 1]
```

Mask is optional but recommended. Without it, the composition step cannot distinguish "stroke painted black here" from "stroke not present here."

### env.step() Interface

```
Input:  action  : Tensor[batch, STROKE_DIM * STROKES_PER_STEP]  # [0, 1]
Output: obs     : Tensor[batch, 7, IMG_SIZE, IMG_SIZE]          # canvas + target + step
        reward  : Tensor[batch, 1]                              # normalized ΔL2
        done    : Tensor[batch, 1]                              # bool
```

### Actor Interface

```
Input:  obs     : Tensor[batch, 7, IMG_SIZE, IMG_SIZE]
Output: action  : Tensor[batch, STROKE_DIM * STROKES_PER_STEP]  # tanh → [0, 1]
```

### Critic Interface

```
Input:  obs     : Tensor[batch, 7, IMG_SIZE, IMG_SIZE]
        action  : Tensor[batch, STROKE_DIM * STROKES_PER_STEP]
Output: q_value : Tensor[batch, 1]
```

### Replay Buffer Interface

```
push(obs, action, reward, next_obs, done)   # stores one transition per call
sample(batch_size) → (obs, action, reward, next_obs, done)
    # each: Tensor[batch_size, ...]
```

### Hard Rasterizer Interface

```
Input:  canvas  : Tensor[batch, 3, H, W]    # uint8 or float [0, 255]
        params  : Tensor[batch, STROKE_DIM]  # floats [0, 1]
Output: canvas  : Tensor[batch, 3, H, W]    # same dtype, rectangle drawn in-place
```

---

## Architectural Decisions and Rationale

### Why R renders the stroke alone (not the full canvas)

If R output the composited canvas, it would need to model all occlusion history — exponentially harder. Rendering only the new stroke isolates R's learning task. Composition is then simple: overwrite pixels where the mask is nonzero. This separation is the right call and matches the paper.

### Why the reward normalizes by initial L2

Raw L2 varies by orders of magnitude across images (solid color target vs. noisy texture). Without normalization, the critic learns image-specific Q-value scales, slowing transfer. Normalized reward keeps all episodes in the same scale, stabilizing training.

### Why target networks with small tau

DDPG is notoriously unstable without target networks. With τ=0.001, target networks change by 0.1% per step — slow enough to give the critic a stable regression target, fast enough to eventually track the improving policy.

### Why 96 parallel environments

Single-env DDPG on image-space states fills the replay buffer extremely slowly. 96 envs run the same batched forward pass on GPU with near-zero overhead. This is the primary lever for training speed and should not be reduced unless VRAM is the bottleneck.

### Why the actor outputs k=5 strokes simultaneously

Each RL step requires an environment interaction. With k=1, a 200-stroke episode = 200 steps, each needing a full actor forward pass and a replay buffer entry. With k=5, the same episode = 40 steps, reducing variance in credit assignment by 5x. The tradeoff is the intra-bundle blind spot (strokes 2–5 don't see canvas updated by stroke 1), which is acceptable for the baseline.

---

## Sources

- Huang et al., "Learning to Paint With Model-Based Deep Reinforcement Learning," ICCV 2019: https://ar5iv.labs.arxiv.org/html/1903.04411
- Reference implementation (hzwer): https://github.com/hzwer/ICCV2019-LearningToPaint
- DDPG algorithm reference: Lillicrap et al., "Continuous Control with Deep Reinforcement Learning" (ICLR 2016)
- Project design document: `paint_ai_design.md` (primary source for this project's specific choices)
