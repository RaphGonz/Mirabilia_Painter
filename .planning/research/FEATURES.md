# Features Research — Paint AI

**Domain:** RL-based stroke-parametric painting (reproduction of target images)
**Reference paper:** "Learning to Paint with Model-Based Deep Reinforcement Learning" (Huang et al., ICCV 2019)
**Researched:** 2026-06-08
**Overall confidence:** HIGH (paper fully available, design decisions locked and documented)

---

## Table Stakes (v1 must-haves)

Features without which the system does not produce a working painting agent. Every item below
is a hard dependency for the training loop to run at all.

| Feature | Why Required | Complexity | Notes |
|---------|--------------|------------|-------|
| Hard rasterizer `draw(canvas, params)` | Ground truth for pretraining R; final clean render at eval time | Low | Pure tensor ops, no gradient needed. STROKE_DIM=8: (cx,cy,w,h,θ,r,g,b). Must produce opaque oriented rectangle. |
| Neural renderer R (pretrain supervised) | Differentiable proxy of rasterizer — lets DDPG actor receive gradients through the stroke render step | Medium | MLP or conv decoder. Trained on (params → stroke image) pairs sampled from the hard rasterizer. Frozen after pretraining. Must reproduce stroke geometry and color faithfully at 64×64. |
| Stroke pretraining dataset generator | R needs a large, uniform random sample of (params, rendered stroke) pairs to train on | Low | Random sampling over the full parameter hypercube. No human data required. |
| RL environment `env.py` (reset / step) | Wraps canvas state, computes transitions, terminates episodes | Low-Medium | State = concat(canvas, target) as 6-channel 64×64 tensor. `step` applies k=5 strokes via R, recomposes canvas, returns reward. Fixed N_STROKES per episode. |
| Incremental L2 reward | Dense reward signal per bundle — critical for credit assignment with opaque strokes | Low | r_t = L2(canvas_{t-1}, target) − L2(canvas_t, target). Computed after the full bundle of k=5 strokes. Positive when canvas improves. |
| Actor network (CNN → continuous action) | Produces stroke parameters from current (canvas, target) observation | Medium | CNN on 6×64×64 state → flat vector of STROKE_DIM × k = 40 continuous values in [0,1]. All k strokes decided simultaneously from the same state. |
| Critic network Q(state, action) | Estimates expected return; provides gradient signal to actor | Medium | Standard DDPG critic. Input: (state, action) concatenated. Output: scalar Q value. |
| Target networks (actor + critic) | Stabilise training by breaking feedback loops in Bellman targets | Low | Soft-update rule τ (e.g. 0.005). Standard DDPG component. Without these, critic diverges. |
| Replay buffer | Breaks temporal correlations in experience samples; enables off-policy learning | Low | Circular buffer of (s, a, r, s') transitions. Typical size 1e5–1e6. Uniform random sampling per batch. |
| Exploration noise | Actor is deterministic — noise needed to explore stroke parameter space during training | Low | Gaussian noise on actor output, decayed over training. OU noise is an alternative but Gaussian is simpler and sufficient. |
| Bundle composition (k=5 strokes per step) | Agent outputs k strokes simultaneously; all must be rendered and composited before reward | Low-Medium | Strokes applied sequentially via R with alpha-compositing (or masked overwrite for opaque). Canvas updated once per bundle. Gradient flows through each stroke in the bundle. |
| Discrete colour palette + nearest-neighbour projection | Agent outputs continuous RGB; projection to ~40-colour palette at eval time | Low | palette.py: array of ~40 RGB values. Projection = argmin L2 in RGB space. Applied post-hoc at inference, not during training. Agent trains on continuous colour. |
| Training loop `train.py` | Drives the interaction between env, agent, and replay buffer; handles checkpointing | Medium | Standard DDPG loop: collect experience → store → sample batch → update critic → update actor via policy gradient through R. Logging loss, reward, L2 per episode. |
| Eval pipeline `eval.py` | Runs a trained agent on a target image, applies palette projection, renders final output via hard rasterizer | Low-Medium | Stops after fixed N_STROKES (or early via L2-gain threshold). Produces final net image by replaying ordered stroke list through hard rasterizer. |
| Frame-by-frame timelapse export | Core deliverable for the Mirabilia series — the "AI painting" video | Low | Save canvas state after each bundle step as PNG. Combine into video (cv2 or ffmpeg). No ML content; pure I/O. |
| Config module `config.py` | Single source of truth for all hyperparams — N_STROKES, STROKE_DIM, IMG_SIZE, k, etc. | Low | Avoids magic numbers scattered across files. Changing resolution or bundle size should be a one-line edit. |

---

## Differentiators (future episodes)

Features that improve painting quality or enable new capabilities but are not required for the
baseline to function. Each one should be introduced as a response to a diagnosed symptom.

| Feature | Value Proposition | Complexity | Trigger Symptom | Notes |
|---------|-------------------|------------|-----------------|-------|
| WGAN adversarial reward | Richer textures, less blurry output. Paper shows WGAN reward achieves lower L2 than L2-reward directly | High | Agent converges to blurry average ("bouillie floue") | Requires discriminator network D, Wasserstein loss, alternating D/actor updates. Major architectural addition. |
| Coarse-to-fine multi-scale | Better coverage of large uniform areas + fine detail; mirrors paper's emergent behaviour at scale | High | 64×64 too coarse for target images; details lost | Divide image into patches; agent per scale. Quasi-mandatory beyond 128×128. |
| Resolution scaling (128×128 → 256×256) | Higher fidelity output; better demo material | Medium | 64×64 demonstrably too low for target content | Larger state tensors; may require deeper actor/critic CNNs. |
| TD3 (Twin Delayed Deep Deterministic Policy Gradient) | Reduces overestimation bias in critic; more stable training | Medium | Q-value divergence, critic instability with DDPG | Drop-in replacement for DDPG. Double critic + delayed actor updates + target smoothing. |
| Gumbel-softmax colour selection | Differentiable discrete palette selection — agent learns palette membership end-to-end | High | Nearest-neighbour projection visibly degrades render quality | Replaces continuous RGB head with categorical over palette. Incompatible with current continuous action space design. |
| Step/depth channel in state | Agent can condition on how far into the episode it is — may improve coarse-to-fine ordering | Low-Medium | Agent treats all steps identically, no progress awareness | Add normalised step counter as an extra state channel (7×H×W). Low risk extension. |
| Stop signal (learned or heuristic) | Agent stops painting when marginal gain per stroke drops below ε | Low (heuristic) / Medium (learned) | Wasteful strokes at end of episode doing no work | Heuristic threshold on L2-gain/stroke is sufficient and already planned. Full learned stop is future. |
| Content-masked loss | Weight reward by regions with semantic content (edges, faces) — more human-like planning | High | Agent wastes strokes on background, ignores salient regions | Requires saliency/edge map as auxiliary input. Changes reward structure significantly. |
| Stroke shape variants (arcs, Bézier) | More expressive mark-making; closer to original paper's arc strokes | Medium | Rectangular strokes feel mechanical, not painterly | Requires retraining R for each new stroke type. Design decision is deliberately locked to rectangles for v1. |

---

## Anti-Features (deliberately excluded from v1)

Things that would add complexity, delay validation of the core loop, or require the baseline
to work first before they are meaningful.

| Anti-Feature | Why Exclude | What to Do Instead |
|--------------|-------------|-------------------|
| WGAN reward from day one | Adds a discriminator training loop, mode collapse risk, and debugging surface before the core RL loop is even verified | Train with L2 reward until convergence is demonstrated, then switch |
| Transparent / alpha strokes | Requires compositional alpha-blending in R; gradient through alpha composition is more complex; paper uses this but our simplification is deliberate | Opaque rectangles — simpler, faster to validate |
| Gumbel-softmax on palette | Incompatible with continuous DDPG action space; adds categorical output head complexity | Nearest-neighbour projection at eval time only |
| Learned stop signal | Requires the agent to already paint well before a stop signal is meaningful; adds extra output head | Fixed N_STROKES with a posteriori L2-gain threshold for early stopping |
| Multi-scale / patch decomposition | Requires a working single-scale agent first; coarse-to-fine is an architectural change, not an incremental addition | Single-scale 64×64 baseline first |
| Human painting data / imitation learning | Adds data collection and supervised pretraining phase; paper does not require it | Random stroke sampling is sufficient for R pretraining |
| Per-stroke canvas update within a bundle | Agent can't see intra-bundle canvas updates by design (locked decision); implementing this would require sequential R calls and breaks the batch gradient flow | All k strokes from the same state; composition is sequential but the actor doesn't see intermediate states |
| Resolution > 64×64 in v1 | Larger tensors, slower iteration, no new insight at baseline stage | 64×64 until baseline produces recognisable output |
| Style transfer / artistic style objectives | Out of scope for reproduction task — the goal is L2 image reconstruction, not aesthetic stylisation | Perceptual loss / style objectives are a separate research direction |
| Real-time interactive painting | No latency requirement for v1; training and eval are offline | Batch eval; timelapse is assembled post-hoc |

---

## Feature Dependencies

```
Hard rasterizer (draw)
    └── Stroke pretraining dataset generator
            └── Neural renderer R (pretrained, frozen)
                    ├── RL environment env.py (uses R for step transitions)
                    │       ├── Incremental L2 reward
                    │       └── Bundle composition (k=5)
                    │               └── Training loop train.py
                    │                       ├── Actor network
                    │                       ├── Critic network
                    │                       ├── Target networks
                    │                       ├── Replay buffer
                    │                       └── Exploration noise
                    └── Eval pipeline eval.py (uses hard rasterizer for final render)
                            ├── Palette projection (nearest-neighbour)
                            └── Timelapse export

Config module (config.py)
    └── depended on by all modules (no ML dependencies itself)
```

**Critical path for v1:**
1. Hard rasterizer — no downstream components work without it
2. Neural renderer R pretraining — RL environment requires it
3. DDPG core (actor + critic + target networks + replay buffer + training loop)
4. Reward + env — wires DDPG to the painting task
5. Eval + timelapse — produces the deliverable

**Independent from core loop (can be built in parallel):**
- Config module
- Palette module
- Timelapse export utility

**Future features that require a working v1 baseline first:**
- WGAN (requires baseline convergence to diagnose blur symptom)
- TD3 (requires instability symptom to justify)
- Coarse-to-fine (requires working single-scale agent)
- Gumbel-softmax (requires visible colour-quantisation degradation to justify)

---

## MVP Recommendation

The v1 MVP is the system as described above under Table Stakes. Every item in that table is
required. No table-stakes feature can be deferred.

**Prioritised build order within Table Stakes:**
1. `config.py` + `palette.py` (no dependencies, define all constants)
2. Hard rasterizer `renderer.py` (ground truth, needed next)
3. `pretrain_renderer.py` + `models/renderer.py` (R, validate visually before touching RL)
4. `env.py` + `reward.py` (wraps R into RL interface)
5. `models/actor.py` + `models/critic.py` (policy networks)
6. `ddpg/replay_buffer.py` + `ddpg/agent.py` (DDPG mechanics)
7. `train.py` (full loop)
8. `eval.py` + timelapse export (deliverable)

**Defer all Differentiators** until the training curve shows convergence and a recognisable
painting is produced. Introducing WGAN, multi-scale, or TD3 before the baseline converges
makes debugging impossible.

---

## Sources

- Huang et al., "Learning to Paint with Model-Based Deep Reinforcement Learning," ICCV 2019 — https://arxiv.org/abs/1903.04411v3
- Official implementation — https://github.com/hzwer/ICCV2019-LearningToPaint
- ar5iv full paper render — https://ar5iv.labs.arxiv.org/html/1903.04411
- Lillicrap et al., "Continuous Control with Deep Reinforcement Learning" (DDPG paper) — referenced for replay buffer and target network details
- Project design document: `paint_ai_design.md` (locked decisions, file structure, evolution plan)
- Project requirements: `.planning/PROJECT.md` (active requirements, out-of-scope list)
