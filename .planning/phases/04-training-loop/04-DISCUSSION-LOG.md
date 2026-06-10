# Phase 4: Training Loop - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-10
**Phase:** 4-Training Loop
**Areas discussed:** Training dataset, Batched env design, Training budget & warm-up, Compositing helper placement

---

## Training Dataset

| Option | Description | Selected |
|--------|-------------|----------|
| CIFAR-10 | 60k images, 32×32 upsampled to 64×64. Dead simple via torchvision. | |
| STL-10 | 13k+100k images, 96×96 downsampled. Richer detail. | |
| CelebA | 200k face images at 64×64. No torchvision autodownload on Windows. | |
| Custom HuggingFace dataset | sezenkarakus/image-description-dataset-v2 — HD images + descriptions, very diverse | ✓ |

**User's choice:** Custom dataset — `sezenkarakus/image-description-dataset-v2`. Preferred for HD quality and diversity.

**Q — Data location / structure:**
User clarified: parquets at `D:\Images\train\parquets`, raw extracted PNGs at `D:\Images\train\raw`. Flat folder, PNG format.

**Q — preserve_data_script:**

| Option | Description | Selected |
|--------|-------------|----------|
| data/train/ + prepare_data.py | Standalone extraction script | ✓ (by user clarification — specific paths given) |
| Inline in train.py | Auto-extract if folder missing | |

**Q — Text descriptions:**

| Option | Description | Selected |
|--------|-------------|----------|
| Images only — drop text | Simple, no overhead | |
| Save descriptions as sidecar CSV | descriptions.csv for future conditioning experiments | ✓ |

**Notes:** User wants descriptions preserved even though they won't be used in Phase 4 training.

---

## Batched Env Design

| Option | Description | Selected |
|--------|-------------|----------|
| Single BatchedPaintEnv | One class, all 96 canvases as (96,3,64,64) tensor, all ops batched | ✓ |
| List of 96 PaintEnv instances | 96 separate objects, loop in train.py | |

**User's choice:** Option A — "on fait comme le papier". User initially unfamiliar with the concept; after explanation in French, confirmed immediately.

**Notes:** User needed clarification that "96 parallel envs" means running 96 painting sessions simultaneously on the GPU, not 96 separate Python processes.

---

## Training Budget & Warm-up

**Q — Total episodes:**

| Option | Description | Selected |
|--------|-------------|----------|
| ~2000 episodes (fast) | Validation run | |
| ~10 000 episodes (solid baseline) | Paper-level training | |
| Configurable via CLI | --max-episodes + config.py param | ✓ |

**User's choice:** Configurable + `MAX_EPISODES` in config.py. Explicit request: autoresearch should use 2000 for quick validation, full run at the end.

**Q — Warm-up:**
User was unfamiliar with the concept. Claude decided: `WARMUP_STEPS = 1000` in config.py (standard value, not discussed further).

**Q — Checkpoints:**

| Option | Description | Selected |
|--------|-------------|----------|
| Every 500 episodes | actor.pt + critic.pt + optimizer states in checkpoints/ | ✓ |
| Only at end | Simpler but lose everything on crash | |

**User's choice:** Every 500 episodes.

---

## Compositing Helper Placement

**Q — Resume from checkpoint:**

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — --resume-from | train.py loads all states and resumes | ✓ |
| No — always from scratch | Simpler | |

**User's choice:** Yes, resume support.

**Notes:** The compositing helper placement itself (`apply_strokes()` location) was decided by Claude — function in `env.py`, imported by agent.py. User is not concerned with code organization details.

---

## Claude's Discretion

- Compositing helper: `apply_strokes()` as module-level function in `env.py` — imported by both env.step() and agent.update_step()
- Warm-up: `WARMUP_STEPS = 1000` (standard value, user didn't need to decide)
- LR scheduler: LinearLR vs manual step — deferred to planner
- Reward edge case: epsilon guard `max(L2_init, 1e-8)` — standard defensive coding

## Deferred Ideas

- Text-conditioned painting (descriptions.csv sidecar preserved for future experiments)
- WGAN discriminator (Phase 6+ scope, triggered by "bouillie floue" symptom)
- Best-checkpoint selection strategy (Phase 5 scope — eval.py picks the checkpoint)
