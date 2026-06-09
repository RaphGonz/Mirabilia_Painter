# Phase 2: Neural Renderer - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-09
**Phase:** 2-Neural Renderer
**Areas discussed:** Compositing strategy, Pretraining budget, Freeze assertion format, R architecture capacity

---

## Compositing strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Soft alpha-blend | alpha = R_out.max(dim=0), new_canvas = alpha * R_out + (1-alpha) * old_canvas. Differentiable end-to-end. | ✓ |
| Hard overwrite where R > threshold | mask = R_out.max(0) > 0.05. Not differentiable at threshold boundary. | |
| Additive clamp | new_canvas = clamp(old_canvas + R_out, 0, 1). Wrong for opaque strokes. | |

**User's choice:** Soft alpha-blend

**Follow-up — dark stroke gap:**

| Option | Description | Selected |
|--------|-------------|----------|
| Accept it — known train/infer gap | Document alongside edge-blending gap already in design doc. Strokes opaque at inference via hard rasterizer replay. | ✓ |
| Use hard rasterizer mask for compositing | mask = draw(zeros, params).max(0) > 0 at each step. Costs one extra rasterizer call per stroke. | |

**User's choice:** Accept the gap

**Notes:** User confirmed the train/infer gap on dark strokes is acceptable — consistent with the design philosophy already documented in paint_ai_design.md.

---

## Pretraining budget

| Option | Description | Selected |
|--------|-------------|----------|
| 500K pairs | ~30-60 min GPU. Faster iteration cycle if R needs retraining. | |
| 1M pairs | ~1-2h GPU. Better coverage of extreme-params distribution. | ✓ |
| Tuned at runtime (CLI arg) | Most flexible, start at 500K and extend. | |

**User's choice:** 1M pairs

**Batch size:**

| Option | Description | Selected |
|--------|-------------|----------|
| 512 | Safe for 8GB+ VRAM, good gradient stability. | |
| 256 | Conservative, lower VRAM for 4-6GB cards. | |
| 1024 | Faster per epoch, needs ~12GB+ VRAM. | ✓ |

**User's choice:** 1024

**Extreme params definition:**

| Option | Description | Selected |
|--------|-------------|----------|
| Thin + tilted + edge | h < 0.05, theta > 0.4, cx/cy < 0.1 or > 0.9. | |
| Full-canvas strokes too | Same as above + w > 0.8 and h > 0.8. | ✓ |

**User's choice:** Include full-canvas strokes in extreme sampling

---

## Freeze assertion format

| Option | Description | Selected |
|--------|-------------|----------|
| pytest test in tests/ | Runs automatically with pytest. Catches accidental re-training. | |
| Inline in pretrain_renderer.py | Runs only when retraining. | |
| Both | Most robust, slightly more code. | |

**User's choice:** Drop entirely — just set requires_grad_(False) correctly

**Notes:** User questioned the value of the assertion for a solo project. After discussing the actual failure mode (forgetting requires_grad_(False) in env.py), user decided the pkl + correct load code is sufficient. The ROADMAP criterion was simplified away from a formal assertion.

---

## R architecture capacity

**Projection path:**

| Option | Description | Selected |
|--------|-------------|----------|
| Linear(8, 512) → reshape (128, 2, 2) → 4 stages | Standard decoder, ~300K params. | ✓ |
| Linear(8, 128) → reshape (32, 2, 2) → 4 stages | Smaller, faster, may struggle with sharp edges. | |

**User's choice:** Linear(8, 512) → reshape (128, 2, 2)

**Filter counts:**

| Option | Description | Selected |
|--------|-------------|----------|
| 128 → 64 → 32 → 16 → 3 | Standard halving, ~300K params. | ✓ |
| 256 → 128 → 64 → 32 → 3 | Doubles capacity, ~1.2M params, slower RL inference. | |
| You decide | Defer to planner. | |

**User's choice:** 128 → 64 → 32 → 16 → 3

---

## Claude's Discretion

- Exact extreme-params thresholds (exact cutoffs for thin/tilted/edge/full-canvas) — planner picks around hard rasterizer subpixel boundary (~0.032 at 64×64)
- Final upsampling strategy for last stage (bilinear upsample + Conv2d vs. stride-2 ConvTranspose2d)

## Deferred Ideas

- Scaling R to 128×128 — future episode
- Perceptual/LPIPS loss — might improve fidelity at higher resolution, not needed at 64×64
- 4-channel output (RGB + explicit alpha mask) — considered, rejected for baseline; revisit if dark-stroke gap causes visible training instability
