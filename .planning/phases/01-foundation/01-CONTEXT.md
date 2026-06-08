# Phase 1: Foundation - Context

**Gathered:** 2026-06-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver the three shared infrastructure modules every downstream phase depends on:
- `config.py` — all project-wide constants
- `palette.py` — the ~40-color manual palette with colorspace-aware projection
- `renderer.py` — the hard rasterizer (opaque oriented rectangle, pure PyTorch tensor ops)

No RL, no neural network, no training loop. This phase is complete when the three modules are importable and correct.

</domain>

<decisions>
## Implementation Decisions

### Episode Structure
- **D-01:** `N_STROKES = 40` steps per episode. Each step applies k=5 strokes, so one episode = 200 total strokes on the canvas. Chosen for 64×64 resolution; gives enough strokes to cover the canvas without excessively long episodes.

### Hard Rasterizer (`renderer.py`)
- **D-02:** Implemented in **pure PyTorch tensor ops** — rotation matrix + meshgrid pixel mask. No cv2 dependency. Runs on GPU, scales to higher resolutions (128×128, 256×256+) without rewrite. The user confirmed intent to scale up resolution in future episodes, which drove this choice over the simpler cv2 approach.
- **D-03:** Angle parameter θ ∈ [0,1] maps to **[0, π]** (half-turn). Rectangles are 180°-symmetric — [0,π] covers all distinct orientations with no redundancy in the actor's action space.
- **D-04:** `draw(canvas, stroke_params)` operates under `torch.no_grad()` — no autograd graph attached to the output. Canvas and output are float32 tensors in [0.0, 1.0].

### Palette (`palette.py`)
- **D-05:** Phase 1 includes the **actual ~40 colors** from the physical paint mixer (not a placeholder). Colors are stored as float [0.0, 1.0] tuples — divide paint mixer's 0–255 values by 255 before entering.
- **D-06:** `project_color(rgb, colorspace)` supports three colorspaces: `"rgb"` (Euclidean L2), `"oklab"` (perceptually uniform), `"hsv"`. Default colorspace is configurable in `config.py`.

### Project File Layout
- **D-07:** **Flat root structure.** All top-level scripts (`config.py`, `palette.py`, `renderer.py`, `env.py`, `train.py`, `eval.py`, `pretrain_renderer.py`) live at project root. Neural models live in `models/` subfolder; DDPG components in `ddpg/` subfolder. No outer `paint_ai/` package directory.
- **D-08:** Import convention: `from config import IMG_SIZE`, `from models.renderer import NeuralRenderer`, etc. No package-relative imports.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Design & Architecture
- `paint_ai_design.md` — Full design decisions table, file structure, two-renderer architecture rationale, points of vigilance (train/inference gap, occlusion order, color projection). Read before implementing any module.

### Requirements
- `.planning/REQUIREMENTS.md` — FOUND-01, FOUND-02, FOUND-03 with exact API signatures, accepted colorspaces, and edge-case coverage requirements for the hard rasterizer. These are the acceptance criteria.

### Reference Paper
- `LearningToPaint.pdf` — Original "Learning to Paint" paper (Huang et al.). Relevant for understanding the two-renderer architecture and the model-based critic design. Not prescriptive for Phase 1 but context for design choices.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- None — greenfield project. No existing Python code.

### Established Patterns
- `paint_ai_design.md` defines the module API contracts: `draw(canvas, stroke_params) → canvas`, `project_color(rgb, colorspace) → palette_rgb`. The planner must match these signatures exactly — they are referenced across all 5 phases.

### Integration Points
- `config.py` is imported by every module in every phase. Keep it free of any non-trivial logic — constants only.
- `renderer.py` (hard) is called by `pretrain_renderer.py` (Phase 2) to generate training data for R, and by `eval.py` (Phase 5) for final rendering. API must be stable.
- `palette.py` `project_color` is called by `eval.py` (Phase 5) at inference time. Colorspace default set in `config.py`.

</code_context>

<specifics>
## Specific Ideas

- User intends to scale resolution to 128×128 and beyond in future episodes (like the paper). The pure-PyTorch rasterizer was chosen explicitly to avoid a rewrite at that point.
- Palette colors are real physical paint colors from a paint mixer — they reflect an actual physical palette, not a computed or algorithmically-generated one.

</specifics>

<deferred>
## Deferred Ideas

- **Resolution scaling (128×128, 256×256+)** — Explicitly a future episode. Drove the rasterizer implementation choice (PyTorch over cv2) but is out of scope here.

</deferred>

---

*Phase: 1-Foundation*
*Context gathered: 2026-06-08*
