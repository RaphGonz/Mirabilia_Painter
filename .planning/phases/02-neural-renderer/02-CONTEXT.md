# Phase 2: Neural Renderer - Context

**Gathered:** 2026-06-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Pre-train neural renderer R from scratch, visually validate its stroke reproduction, freeze it, and verify the freeze is respected in all downstream code.

Deliverables:
- `models/renderer.py` — CNN decoder, `(batch, 8) → (batch, 3, 64, 64)`, no BatchNorm
- `pretrain_renderer.py` — supervised training loop against the hard rasterizer; saves `renderer.pkl`
- Visual gate: human inspection of R output vs. hard rasterizer on thin/tilted/edge/full-canvas strokes

Phase ends when R is frozen and produces recognizable, non-smeared rectangles. Phase 3 (DDPG Models) cannot start until this gate is passed.

Out of scope: RL environment, actor, critic, training loop, evaluation, palette projection.

</domain>

<decisions>
## Implementation Decisions

### Compositing (env.py — Phase 3/4 forward-reference)
- **D-01:** Soft alpha-blend: `alpha = R_out.max(dim=0)`, `new_canvas = alpha * R_out + (1 - alpha) * old_canvas`. Fully differentiable — gradient flows into R during actor backprop.
- **D-02:** Dark-stroke train/infer gap accepted. Strokes with low max-RGB (e.g., deep blue) will appear semi-transparent during RL training; they are fully opaque at inference (hard rasterizer replay). Documented alongside the existing edge-blending gap in `paint_ai_design.md`.
- **D-03:** R training target = `draw(zeros_canvas, params)` — stroke rendered on a black canvas. No compositing in the training objective; R learns the stroke image alone.

### Pretraining budget
- **D-04:** 1M training pairs, generated on-the-fly (no static dataset).
- **D-05:** Batch size 1024. Expected GPU time: ~1-2h on a mid-range NVIDIA card.
- **D-06:** Extreme-params biasing — 20% of each batch samples from extreme regions: thin (h < 0.05 or w < 0.05), tilted (theta_01 > 0.4, i.e., > 72°), edge-position (cx < 0.1 or cx > 0.9 or cy < 0.1 or cy > 0.9), full-canvas (w > 0.8 and h > 0.8). Exact thresholds tunable by planner around the hard rasterizer subpixel boundary (~0.032 at 64×64).
- **D-07:** Validation set: held-out random stroke params (separate from training). Target: val MSE < 0.005. ReduceLROnPlateau on val MSE.

### Freeze
- **D-08:** No formal assertion or pytest test for the freeze. Correct freeze = `model.eval()` + `requires_grad_(False)` on all R parameters when loading `renderer.pkl` in `env.py`. One comment explaining why. This is a solo project — the pkl + correct load code is sufficient.

### R architecture
- **D-09:** Projection path: `Linear(8, 512)` → reshape to `(128, 2, 2)` → 4 conv+upsample stages → `(3, 64, 64)`. Bilinear upsampling doubles spatial resolution at each stage: 2×2 → 4×4 → 8×8 → 16×16 → 64×64 (final stage uses stride-2 conv or a single larger upsample step).
- **D-10:** Filter counts: 128 → 64 → 32 → 16 → 3. Each stage: bilinear upsample + `Conv2d` + `ReLU`. Final projection: `Conv2d(16, 3, 1)` + `Sigmoid` for [0,1] output. ~300K params.
- **D-11:** No BatchNorm anywhere (interacts poorly with single-sample inference in env.py). No Dropout.
- **D-12:** Optimizer: Adam lr=1e-3 with `ReduceLROnPlateau` on val MSE. Gradient clipping not needed for R (it's supervised, not RL-scale instability).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Design & Architecture
- `paint_ai_design.md` — Two-renderer architecture rationale, compositing decision ("R outputs stroke alone, composition outside the network"), train/infer gap documentation, points of vigilance (soft blending, occlusion order). Read before implementing any compositing logic.

### Existing Implementation (Phase 1)
- `renderer.py` — Hard rasterizer API: `draw(canvas, stroke_params) → canvas`. This is R's supervision oracle. Training loop calls `draw(zeros_canvas, params)` to generate targets. Param encoding: `(cx, cy, w, h, theta_01, r, g, b)` all in [0,1]; theta_01 maps to [0,π].
- `config.py` — `IMG_SIZE=64`, `STROKE_DIM=8`, `STROKES_PER_STEP=5`, `N_STROKES=40`.

### Requirements
- `.planning/REQUIREMENTS.md` — REND-01, REND-02, REND-03: exact acceptance criteria for R architecture, pretraining targets, visual gate, and freeze verification requirements.
- `.planning/ROADMAP.md` §Phase 2 — 5 success criteria (architecture shape, val MSE target, visual gate, freeze, hard gate). Planner must satisfy all 5.

### Reference Paper
- `LearningToPaint.pdf` — Original paper's neural renderer design (for architectural reference; this project uses rectangles/opacity, not the paper's brush strokes).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `renderer.py::draw(canvas, stroke_params)` — used as the supervision oracle for R's training data generation. Call as `draw(torch.zeros(3, IMG_SIZE, IMG_SIZE), params)` to get stroke-on-black target.
- `config.py::IMG_SIZE`, `STROKE_DIM` — import directly; no magic numbers in `models/renderer.py` or `pretrain_renderer.py`.

### Established Patterns
- D-08 from Phase 1: `@torch.no_grad()` on the hard rasterizer. R is trained WITH autograd; do NOT decorate R's forward with `@torch.no_grad()` — gradients must flow during pretraining.
- D-07 from Phase 1: flat root structure. `models/renderer.py` lives in `models/`. Import as `from models.renderer import NeuralRenderer`.
- D-11 from Phase 1: import convention `from config import IMG_SIZE`, not package-relative.

### Integration Points
- `env.py` (Phase 4) will load `renderer.pkl` and call R's forward at every `env.step()`. R must be fast at inference (single stroke, no batch overhead).
- `pretrain_renderer.py` is a standalone script (not imported by anything). CLI-launchable: `python pretrain_renderer.py`.
- `models/__init__.py` already exists (empty) — `models/renderer.py` can be added directly.

</code_context>

<specifics>
## Specific Ideas

- R's output is used with soft alpha compositing in env.py: `alpha = R_out.max(dim=0)`, `new_canvas = alpha * R_out + (1-alpha) * old_canvas`. The planner must document this formula in env.py as a comment so the Phase 4 executor knows exactly what to implement.
- Training data is generated on-the-fly (no disk save). Each batch: sample `params` uniform [0,1]^8, call `draw(zeros, params)` for the target, forward R, compute MSE loss, backward.
- Extreme-params biasing: in each batch, randomly select 20% of samples and re-sample their params from the extreme distribution instead of uniform.

</specifics>

<deferred>
## Deferred Ideas

- **Scaling R to 128×128** — architecture is designed to scale (bilinear upsample chain), but resolution increase is a future episode.
- **Perceptual loss for R** — MSE is sufficient at 64×64; perceptual/LPIPS loss could improve fidelity at higher res.
- **R outputting 4 channels (RGB + alpha mask)** — considered and rejected for baseline; revisit if dark-stroke train/infer gap causes visible training instability.

</deferred>

---

*Phase: 2-Neural Renderer*
*Context gathered: 2026-06-09*
