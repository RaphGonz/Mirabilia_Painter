# Phase 3: DDPG Models - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-10
**Phase:** 3-DDPG Models
**Areas discussed:** Actor state channels, Actor CNN architecture, Critic architecture, Replay buffer design

---

## Actor state channels

| Option | Description | Selected |
|--------|-------------|----------|
| 6 channels (canvas + target) | canvas (3) + target (3). CLAUDE.md documented this. | |
| 7 channels (canvas + target + step) | canvas (3) + target (3) + step_progress (1). ROADMAP documented this. | |
| 7 channels (canvas + target + L2 map) | Per-pixel L2 error as spatial channel | |

**User's choice:** Verified against paper — 7 channels confirmed
**Notes:** User requested paper verification instead of trusting CLAUDE.md. Paper §3.2 explicitly: "s_t = (C_t, I, t) — step number acts as additional information to instruct the agent the remaining number of steps." Paper appendix Fig. 13 & 14 both show [C, I, #Step] = 7-channel input for BOTH actor and critic. CLAUDE.md was wrong. Correction also applied to ROADMAP (critic was listed as 6ch — also wrong).

---

## Actor CNN architecture

| Option | Description | Selected |
|--------|-------------|----------|
| Simple 4-layer CNN | No residual connections, fastest to train | |
| Mini ResNet (4 residual blocks) | ResNet-inspired, skip connections | Initial choice |
| ResNet18 + CoordConv (paper) | Full paper architecture | ✓ |

**User's choice:** ResNet18 + CoordConv (paper) — full complexity
**Notes:** User initially selected mini-ResNet, then after paper cross-check and seeing full architecture, decided to go full complexity. "Mon ordi à les capacités de tanker, plus que je ne croyais." CoordConv adds explicit spatial position channels before the first convolution.

**BatchNorm in actor:**
| Option | Description | Selected |
|--------|-------------|----------|
| BatchNorm in actor | Paper confirms BN on actor. Actor not called single-sample at inference. | ✓ |
| GroupNorm | Works at any batch size, deviates from paper | |
| No normalization | Simplest, CLAUDE.md recommendation | |

**User's choice:** BatchNorm in actor — follows paper §3.4 exactly.

---

## Critic architecture

| Option | Description | Selected |
|--------|-------------|----------|
| Same mini-ResNet as actor | Consistent architecture, different final FC | |
| Same mini-ResNet (lighter) | Lighter 3-conv backbone | |

**Final decision:** Same ResNet18 + CoordConv backbone as actor (after full-complexity pivot)

**Critic normalization:**
| Option | Description | Selected |
|--------|-------------|----------|
| No normalization | Simplest, skip WN+TReLU | |
| Weight Normalization | Follows paper exactly | ✓ |
| BatchNorm | Paper explicitly avoids this on critic | |

**User's choice:** Weight Normalization — paper §3.4: "BN can not speed up the critic learning significantly; use WN with TReLU."

**Step channel encoding:**
| Option | Description | Selected |
|--------|-------------|----------|
| Scalar broadcast to 64×64 | t/N_STROKES tiled to full spatial grid | ✓ |
| Extra FC input after CNN | Step as raw scalar appended post-CNN | |

**User's choice:** Broadcast to spatial grid — consistent treatment with other channels.

---

## Replay buffer design

| Option | Description | Selected |
|--------|-------------|----------|
| Store target redundantly per transition | Simple, no deduplication | ✓ |
| Store target per episode | Memory-efficient, complex sampling | |

**User's choice:** Redundant per-transition storage — simpler implementation.

**Buffer implementation:**
| Option | Description | Selected |
|--------|-------------|----------|
| Numpy ring buffer | Pre-allocated, O(1) insert/sample | ✓ |
| Python deque | Simpler code, slower sampling | |

**User's choice:** Numpy ring buffer — standard practice (SB3, CleanRL).

**Buffer size:**
| Option | Description | Selected |
|--------|-------------|----------|
| 800 episodes (~32k, paper) | Paper default for CelebA/ImageNet | |
| 200k transitions | 6× larger, deliberate deviation | ✓ |
| 1000 episodes (~40k) | Slight deviation from paper | |

**User's choice:** 200k transitions — deliberate deviation for MS COCO dataset diversity.
**Notes:** User reasoning: "La base de données va être différente (photos random de type MS COCO) donc il faut plus d'exemple." Larger buffer = more image diversity in training samples, less temporal correlation.

---

## Deferred Ideas

- WGAN discriminator — Phase 4+ evolution when "bouillie floue" symptom appears
- LR scheduler implementation — deferred to Phase 4 (not needed in model definitions)
- CoordConv radius channel — paper doesn't use it, skip for baseline
