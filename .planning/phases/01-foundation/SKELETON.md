# Walking Skeleton — Paint AI (Mirabilia Episode 1)

**Created:** 2026-06-08
**Phase:** 01-foundation
**Status:** Architectural baseline for all downstream phases

> This is an ML/Python project (DDPG painting agent), not a web app. The Walking
> Skeleton is the thinnest end-to-end stack that proves the foundation works:
> the three shared modules import together in one Python session and the hard
> rasterizer draws a recognizable oriented rectangle. Subsequent phases build on
> the decisions recorded here without re-litigating them.

---

## Capability Proven End-to-End

After Phase 1, in a single Python session:

```python
import config            # constants load, zero side effects
import palette           # project_color works in rgb / oklab / hsv
import renderer          # draw() available
import torch
canvas = torch.zeros(3, config.IMG_SIZE, config.IMG_SIZE)
params = torch.tensor([0.5, 0.5, 0.4, 0.1, 0.25, 1., 0., 0.])
canvas = renderer.draw(canvas, params)   # a recognizable red tilted rectangle
nearest = palette.project_color((0.9, 0.1, 0.1))  # nearest palette color
```

- All three modules import with **no circular dependency**.
- `renderer.draw()` output has **no autograd graph** (frozen ground truth for Phase 2).
- A real computation runs: one stroke drawn, one color projected.
- `pytest tests/ -v` is green.

---

## Stack Touched

| Layer | Choice | File(s) |
|-------|--------|---------|
| Scaffold | flat-root layout + pytest | `pyproject.toml`, `tests/` |
| Constants | single-source-of-truth module, zero logic | `config.py` |
| Palette | manual RGB list + colorspace-aware NN projection | `palette.py` |
| Rasterizer | pure PyTorch rotation-matrix + meshgrid mask | `renderer.py` |
| Real computation | draw one oriented rectangle; project one color | `renderer.draw`, `palette.project_color` |
| Tests | unit tests for all three modules + import DAG check | `tests/test_*.py` |

---

## Architectural Decisions (locked for downstream phases)

These are recorded so Phases 2–5 build on them without renegotiating. Source: CONTEXT.md D-01..D-08, RESEARCH.md, CLAUDE.md.

### Project layout — flat root (D-07, D-08)

```
episode1/code/                  (project root = import root)
├── config.py                   # constants only
├── palette.py                  # PALETTE + project_color
├── renderer.py                 # hard rasterizer draw()
├── pyproject.toml              # [tool.pytest.ini_options] testpaths=["tests"]
├── models/__init__.py          # Phase 2+ (NeuralRenderer, Actor, Critic)
├── ddpg/__init__.py            # Phase 3+ (agent, replay_buffer)
└── tests/
    ├── test_config.py
    ├── test_palette.py
    ├── test_renderer.py
    └── test_imports.py
```

- **No outer `paint_ai/` package directory.** Top-level scripts live at root.
- **Absolute flat imports only:** `from config import IMG_SIZE`, `from models.renderer import NeuralRenderer`. Never relative (`from .config`) or package-prefixed (`import paint_ai.config`).
- **Future files** (also flat root): `env.py`, `train.py`, `eval.py`, `pretrain_renderer.py`. Neural nets in `models/`, DDPG components in `ddpg/`.

### Constants ownership (FOUND-01)

- `config.py` has **zero imports and zero logic**. Every module in every phase imports from it. Any derived value (e.g. `TOTAL_STROKES = N_STROKES * STROKES_PER_STEP`) is computed in the caller, never in config.
- Locked values: `IMG_SIZE=64`, `STROKE_DIM=8`, `STROKES_PER_STEP=5`, `N_STROKES=40` (40 steps × 5 strokes = 200 strokes/episode per D-01), `IMAGE_RANGE=(0.0, 1.0)`, `PALETTE_COLORSPACE="rgb"`.

### Hard rasterizer — pure PyTorch (D-02, D-03, D-04)

- **Pure PyTorch tensor ops**, no cv2. Rotation matrix + meshgrid pixel mask. Runs on GPU, scales to higher resolution without rewrite (resolution scaling is a future episode — out of scope here but drove this choice).
- **API:** `draw(canvas, stroke_params) -> canvas`, where `stroke_params` is `(cx, cy, w, h, θ, r, g, b)`, all in `[0,1]`. This signature is stable and consumed by `pretrain_renderer.py` (Phase 2) and `eval.py` (Phase 5).
- **θ mapping:** `θ ∈ [0,1]` maps to `[0, π]` (half-turn). Rectangles are 180°-symmetric, so `[0,π]` covers all orientations with no redundancy in the actor's action space.
- **No autograd:** `@torch.no_grad()` decorator is mandatory. The hard rasterizer is the non-differentiable ground truth that Phase 2's neural renderer R is trained against.
- **Subpixel behavior:** strokes with `w < 2/(IMG_SIZE-1)` (~0.032 at 64×64) cover zero pixels and return the canvas unchanged — correct, not a bug. Phase 2 must bias 20% of training data toward minimum-visible params.
- **Always** `torch.meshgrid(ys, xs, indexing='ij')` and extract scalar θ via `.item()` before `math.cos/sin`.

### Palette + projection (FOUND-02, D-05, D-06)

- Palette is a **manually edited RGB list** (physical paint-mixer colors, stored as float `[0,1]` tuples = uint8/255). Phase 1 ships a clearly-marked **placeholder** of 5–6 colors; the user fills in the real ~40 later. `project_color` is palette-size-agnostic.
- `PALETTE` tensor is built **once at module load**, shape `(P, 3)` float32 — never converted per call.
- **API:** `project_color(rgb, colorspace="rgb") -> palette_rgb` (shape `(3,)`). Supports `"rgb"` (L2), `"oklab"` (perceptual), `"hsv"`. Default colorspace lives in `config.PALETTE_COLORSPACE`. Consumed by `eval.py` (Phase 5) at inference.
- **Nearest-neighbor via `torch.cdist`** (vectorized, GPU-compatible); query `.unsqueeze(0)` to satisfy the 2D shape contract.
- **oklab NaN guard:** `.clamp(min=0.0)` before `.pow(1/3)` — LMS channels go slightly negative near pure black under float32.

### Dependency DAG (no cycles)

```
config.py  (no imports)
   ▲   ▲
   │   └──── renderer.py  (from config import IMG_SIZE)
   └──────── palette.py   (from config import PALETTE_COLORSPACE)
```

`config` imports nothing; `palette` and `renderer` import only `config`. No module imports `palette` or `renderer` at module level in Phase 1. Enforced by `tests/test_imports.py`.

### Environment

- **Python 3.14.3** installed (CLAUDE.md recommends 3.11; PyTorch 2.10.0+cu126 verified working on 3.14 for Phase 1 primitives). Fallback if a later phase hits 3.14 incompatibility: `conda create -n paint_ai python=3.11`.
- **PyTorch 2.10.0+cu126**, CUDA 12.6, GTX 1660 Ti. No new packages installed in Phase 1.
- **Testing:** pytest 8.4.2, `testpaths=["tests"]`, `pytest tests/ -x -q` (~5s).

---

## What This Skeleton Deliberately Does NOT Include

- No RL, no neural network, no training loop (Phases 2–5).
- No real palette colors yet (placeholder; user fills in physical paint-mixer values).
- No resolution scaling beyond 64×64 (future episode; rasterizer is written to support it without rewrite).
- No WGAN, TD3, Gumbel-softmax, learned stop signal (v2 / out of scope per REQUIREMENTS.md).
