# Phase 1: Foundation - Research

**Researched:** 2026-06-08
**Domain:** PyTorch tensor ops, colorspace math, Python project structure
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** `N_STROKES = 40` steps per episode. Each step applies k=5 strokes, so one episode = 200 total strokes on the canvas.
- **D-02:** Hard rasterizer implemented in **pure PyTorch tensor ops** — rotation matrix + meshgrid pixel mask. No cv2 dependency. Runs on GPU, scales to higher resolutions without rewrite.
- **D-03:** Angle parameter θ ∈ [0,1] maps to **[0, π]** (half-turn). Rectangles are 180°-symmetric — [0,π] covers all distinct orientations.
- **D-04:** `draw(canvas, stroke_params)` operates under `torch.no_grad()` — no autograd graph attached to the output. Canvas and output are float32 tensors in [0.0, 1.0].
- **D-05:** Phase 1 includes the **actual ~40 colors** from the physical paint mixer. Colors stored as float [0.0, 1.0] tuples.
- **D-06:** `project_color(rgb, colorspace)` supports three colorspaces: `"rgb"`, `"oklab"`, `"hsv"`. Default colorspace configurable in `config.py`.
- **D-07:** **Flat root structure.** `config.py`, `palette.py`, `renderer.py` at root. `models/` and `ddpg/` as subfolders.
- **D-08:** Import convention: `from config import IMG_SIZE`, `from models.renderer import NeuralRenderer`.

### Claude's Discretion

None explicitly listed for Phase 1. All design decisions are locked.

### Deferred Ideas (OUT OF SCOPE)

- **Resolution scaling (128×128, 256×256+)** — Future episode. Rasterizer must work at any resolution but this is not tested in Phase 1.

</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FOUND-01 | `config.py` exposes `IMG_SIZE=64`, `STROKE_DIM=8`, `STROKES_PER_STEP=5`, `N_STROKES`, `IMAGE_RANGE=(0.0, 1.0)` importable from all modules | Constants-only module, no logic; no circular dep risk |
| FOUND-02 | `palette.py` defines manual RGB list and `project_color(rgb, colorspace) → palette_rgb` with `colorspace ∈ {"rgb", "oklab", "hsv"}` | `torch.cdist` for L2; exact oklab and hsv formulas verified in-session |
| FOUND-03 | `renderer.py` exposes `draw(canvas, stroke_params) → canvas` — opaque oriented rectangle via pure tensor ops, no autograd | Rotation matrix + meshgrid mask algorithm verified working on GPU in-session |

</phase_requirements>

---

## Summary

Phase 1 delivers three pure Python/PyTorch modules with no external dependencies beyond PyTorch itself. The work is algorithmic, not architectural: the hard rasterizer is a rotation-matrix plus meshgrid pixel test; palette projection is a `torch.cdist` nearest-neighbor search; colorspace conversion is a sequence of closed-form matrix operations. All three algorithms were verified working in the current environment (PyTorch 2.10.0+cu126, Python 3.14, GTX 1660 Ti) during this research session.

The biggest implementation risk is the oriented rectangle rasterizer edge-case handling. A rectangle with width or height below ~2/63 (~0.032 in [0,1] space) will cover zero pixels at 64×64 resolution, which is correct behavior for a hard rasterizer but must be explicitly documented so the neural renderer pre-trainer knows to bias training data away from degenerate strokes. The algorithm itself is correct for all valid inputs.

okLab conversion requires a sRGB gamma linearization step before the matrix transforms. The exact formulas were fetched from Björn Ottosson's canonical page and verified in-session. HSV conversion is branch-free with `torch.where` operations.

**Primary recommendation:** Implement `config.py` first (trivial), then `palette.py` (math only, no GPU), then `renderer.py` (GPU-compatible tensor ops). All three are standalone with no interdependencies except `palette.py` and `renderer.py` importing `config.py`.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Project constants (`IMG_SIZE`, `N_STROKES`, etc.) | config.py (single source of truth) | — | Every downstream module imports this; must have zero logic |
| Hard rasterizer (`draw`) | renderer.py (pure computation) | — | Called by pretrain_renderer.py (Phase 2) and eval.py (Phase 5); must be stable API |
| Palette storage + projection (`project_color`) | palette.py | config.py (default colorspace) | `project_color` called at inference time by eval.py; colorspace default lives in config |
| Colorspace math (oklab, hsv) | Inside palette.py | — | Not exposed as public API; only used internally by `project_color` |

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.14.3 (installed) | Runtime | CLAUDE.md recommends 3.11 but 3.14 is installed and PyTorch 2.10 works on it [VERIFIED: in-session test] |
| PyTorch | 2.10.0+cu126 (installed) | Tensor ops, `torch.meshgrid`, `torch.cdist`, `torch.where` | All Phase 1 ops verified working [VERIFIED: in-session test] |
| math (stdlib) | — | `math.pi`, `math.cos`, `math.sin` for scalar θ | No dependency; faster than `torch.cos` for scalar angle |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | 8.4.2 (installed) | Unit tests for all three modules | Run per-task for shape assertions and edge cases |
| numpy | 2.3.5 (installed) | Not needed in Phase 1 | Do not import numpy in config/palette/renderer — keep pure PyTorch |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Pure PyTorch rasterizer | cv2 `fillPoly` | cv2 is CPU-only and would require uint8/BGR roundtrip; locked out by D-02 |
| `torch.cdist` for palette NN | Manual loop | Manual loop is O(N×P) Python loop; cdist is vectorized and GPU-compatible |
| `math.cos/sin` for θ | `torch.cos/sin` | Scalar θ from stroke_params[4] can use math functions; avoids device mismatch risk when computing cos/sin outside tensor graph |

**Installation:** All required packages are already installed. No `pip install` needed for Phase 1.

---

## Package Legitimacy Audit

Phase 1 installs zero new packages. All dependencies (torch, torchvision, numpy, pytest, matplotlib) are already present in the environment.

| Package | Registry | Installed Version | slopcheck | Disposition |
|---------|----------|-------------------|-----------|-------------|
| torch | PyPI | 2.10.0+cu126 | [OK] | Approved |
| numpy | PyPI | 2.3.5 | [OK] | Approved |
| pytest | PyPI | 8.4.2 | [OK] | Approved |
| matplotlib | PyPI | 3.10.8 | [OK] | Approved |
| torchvision | PyPI | 0.25.0+cu126 | [OK] | Approved |
| tqdm | PyPI | 4.67.3 | [OK] | Approved |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

*slopcheck 0.6.1 ran successfully on all 6 packages; all rated [OK].*

---

## Architecture Patterns

### System Architecture Diagram

```
stroke_params (8 floats, [0,1])
        |
        v
   renderer.py
   draw(canvas, stroke_params)
        |
        |-- meshgrid(H, W) -> (grid_y, grid_x)
        |-- translate by (cx, cy)
        |-- rotate by theta*pi via 2x2 rotation matrix
        |-- half-plane test: |local_x| <= w/2 AND |local_y| <= h/2
        |-- torch.where(mask, color, canvas)
        v
   canvas (3, H, W) float32 [0,1]

palette_colors (list of ~40 RGB tuples)
        |
        v
   palette.py
   project_color(rgb, colorspace)
        |
        |-- [oklab] sRGB linearize -> M1 -> cbrt -> M2
        |-- [hsv]   rgb_to_hsv via torch.where branches
        |-- [rgb]   identity
        |-- torch.cdist(query, palette_converted) -> argmin
        v
   nearest palette color (RGB float32 [0,1])
```

### Recommended Project Structure

```
paint_ai/ (project root)
├── config.py              # constants only — no imports, no logic
├── palette.py             # PALETTE list + project_color(); imports config, torch, math
├── renderer.py            # draw(); imports config, torch, math
├── models/                # Phase 2+
│   └── __init__.py
├── ddpg/                  # Phase 3+
│   └── __init__.py
└── tests/
    ├── test_config.py     # import + value assertions
    ├── test_palette.py    # project_color for all 3 colorspaces
    └── test_renderer.py   # draw() shape, edge cases, no-autograd check
```

### Pattern 1: Rotation-Matrix Pixel Mask (Hard Rasterizer)

**What:** Build a coordinate grid for all pixels, rotate each pixel's offset into the rectangle's local frame, then test whether it falls within the rectangle's half-extents.

**When to use:** Any opaque rectangle rendering without anti-aliasing.

**Why rotation matrix over affine_grid/grid_sample:** `torch.nn.functional.affine_grid` + `grid_sample` is designed for differentiable spatial transformers (requires float theta, output is bilinear interpolated). For a hard opaque mask we want exact boolean pixel assignment — the rotation matrix approach is direct, readable, and verified faster for this use case.

**Example:**
```python
# Source: in-session verification (2026-06-08)
import torch, math

@torch.no_grad()
def draw(canvas: torch.Tensor, stroke_params: torch.Tensor) -> torch.Tensor:
    """
    Draw an opaque oriented rectangle onto canvas.

    Args:
        canvas: float32 tensor (3, H, W) in [0, 1]
        stroke_params: float32 tensor (8,) = (cx, cy, w, h, theta_01, r, g, b)
                       all values in [0, 1]; theta_01 maps to [0, pi]

    Returns:
        New canvas tensor (3, H, W), no autograd graph.
    """
    cx, cy, w, h, theta_01, r, g, b = stroke_params
    theta = theta_01.item() * math.pi          # scalar, avoids device mismatch

    H_px, W_px = canvas.shape[-2], canvas.shape[-1]
    device = canvas.device

    # Pixel coordinate grid in [0, 1] x [0, 1]
    # linspace(0, 1, N): spacing = 1/(N-1), pixel centers at 0, 1/(N-1), ... 1
    ys = torch.linspace(0.0, 1.0, H_px, device=device)
    xs = torch.linspace(0.0, 1.0, W_px, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W) each

    # Translate to rectangle-center-relative coordinates
    dx = grid_x - cx
    dy = grid_y - cy

    # Rotate pixel offsets into rectangle's local frame
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    local_x =  cos_t * dx + sin_t * dy   # along rectangle's length axis
    local_y = -sin_t * dx + cos_t * dy   # along rectangle's width axis

    # Boolean mask: pixel is inside rectangle
    mask = (local_x.abs() <= w / 2.0) & (local_y.abs() <= h / 2.0)  # (H, W) bool
    mask = mask.unsqueeze(0)  # (1, H, W) for broadcast with (3, H, W) canvas

    # Paint: broadcast color over all pixels, select with mask
    color = torch.stack([r, g, b]).view(3, 1, 1).expand_as(canvas)
    return torch.where(mask, color, canvas)
```

**Critical note on `indexing='ij'`:** Always pass `indexing='ij'` to `torch.meshgrid`. Without it, PyTorch 2.10 emits a `UserWarning` about upcoming default change. With `indexing='ij'`, the first output grid varies along dim-0 (rows/y) and the second along dim-1 (cols/x). [VERIFIED: in-session test]

### Pattern 2: Colorspace-Aware Nearest-Neighbor Projection

**What:** Convert both query and palette to the target colorspace, then use `torch.cdist` to find the nearest palette entry.

**When to use:** Any palette lookup in `project_color`.

**Example:**
```python
# Source: in-session verification + bottosson.github.io/posts/oklab/ (2026-06-08)
import torch, math

def _srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    """Apply inverse sRGB gamma to get linear light values. [VERIFIED: sRGB spec]"""
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def _linear_to_oklab(rgb_linear: torch.Tensor) -> torch.Tensor:
    """
    Convert linear sRGB to oklab.
    rgb_linear: (..., 3) in [0, 1] linear light.
    Returns: (..., 3) [L, a, b]
    Source: https://bottosson.github.io/posts/oklab/
    """
    r, g, b = rgb_linear[..., 0], rgb_linear[..., 1], rgb_linear[..., 2]
    l = 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
    m = 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
    s = 0.0883024619*r + 0.2817188376*g + 0.6299787005*b
    # Cube root — clamp first to avoid NaN on tiny negatives from float error
    l_ = l.clamp(min=0.0).pow(1.0 / 3.0)
    m_ = m.clamp(min=0.0).pow(1.0 / 3.0)
    s_ = s.clamp(min=0.0).pow(1.0 / 3.0)
    L =  0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_
    a =  1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_
    b_ = 0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_
    return torch.stack([L, a, b_], dim=-1)

def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """Convert sRGB to HSV. rgb: (..., 3) in [0, 1]. Returns (..., 3) [H, S, V]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = torch.max(rgb, dim=-1).values
    cmin = torch.min(rgb, dim=-1).values
    delta = cmax - cmin
    v = cmax
    s = torch.where(cmax > 0, delta / cmax, torch.zeros_like(cmax))
    h = torch.zeros_like(cmax)
    h = torch.where((cmax == r) & (delta > 0), ((g - b) / delta) % 6.0, h)
    h = torch.where((cmax == g) & (delta > 0), (b - r) / delta + 2.0, h)
    h = torch.where((cmax == b) & (delta > 0), (r - g) / delta + 4.0, h)
    h = h / 6.0
    return torch.stack([h, s, v], dim=-1)

def project_color(
    rgb: tuple | list | torch.Tensor,
    palette: torch.Tensor,
    colorspace: str = "rgb"
) -> torch.Tensor:
    """
    Return the nearest palette color to `rgb` in the given colorspace.

    Args:
        rgb: sRGB color, shape (3,) or length-3 sequence, values in [0, 1]
        palette: tensor (P, 3) of palette colors in sRGB [0, 1]
        colorspace: one of "rgb", "oklab", "hsv"

    Returns:
        Nearest palette color as tensor (3,) in sRGB [0, 1]
    """
    q = torch.as_tensor(rgb, dtype=torch.float32).unsqueeze(0)  # (1, 3)
    pal = palette.float()

    if colorspace == "oklab":
        q_conv = _linear_to_oklab(_srgb_to_linear(q))
        pal_conv = _linear_to_oklab(_srgb_to_linear(pal))
    elif colorspace == "hsv":
        q_conv = _rgb_to_hsv(q)
        pal_conv = _rgb_to_hsv(pal)
    else:  # "rgb" — Euclidean L2 in sRGB
        q_conv = q
        pal_conv = pal

    dists = torch.cdist(q_conv, pal_conv)   # (1, P)
    idx = dists.argmin(dim=-1).item()
    return palette[idx]
```

### Anti-Patterns to Avoid

- **Using `torch.meshgrid` without `indexing='ij'`:** Causes a `UserWarning` in PyTorch 2.10 and will silently break in a future release when the default flips to `'xy'`. Always pass `indexing='ij'` explicitly.
- **Calling `math.cos/sin` on a tensor:** If `theta_01` is a tensor, calling `math.cos(theta_01)` raises a TypeError. Extract the scalar with `.item()` first: `theta_val = theta_01.item() * math.pi`.
- **Using `torch.cos/sin` for `theta` and then doing non-tensor arithmetic mixing:** Either stay fully in tensors or fully in Python floats for angle computation. The recommended pattern above uses `.item()` to go scalar immediately after extracting `theta_01`.
- **Storing `PALETTE` as a plain Python list in palette.py and converting on every call:** Convert once at module load with `PALETTE = torch.tensor([...], dtype=torch.float32)`.
- **Importing `config.py` via relative imports:** Use `from config import IMG_SIZE` — absolute flat imports per D-08. No `from .config import` or `import paint_ai.config`.
- **Adding logic to `config.py`:** It must be constants only. Any computation (e.g., deriving `TOTAL_STROKES = N_STROKES * STROKES_PER_STEP`) belongs in the caller, not in config.
- **Attaching autograd to rasterizer output:** The `@torch.no_grad()` decorator on `draw()` guarantees no gradient graph. Do not remove it — downstream neural renderer training specifically uses the hard rasterizer as non-differentiable ground truth.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Nearest-neighbor search in colorspace | Manual Python loop over palette | `torch.cdist(query, palette).argmin()` | Vectorized, GPU-compatible, O(1) PyTorch call vs O(P) Python overhead |
| sRGB gamma correction | Custom approximation | Exact IEC 61966-2-1 piecewise formula (see Code Examples) | Approximations (e.g., simple `x^(1/2.2)`) give wrong lightness; neural renderer training depends on accurate color distances |
| Rotation matrix | Custom trig lookup | `math.cos`, `math.sin` directly | Standard; no hand-rolled approximation needed |

**Key insight:** For Phase 1, "don't hand-roll" mostly means "don't loop in Python where PyTorch tensor ops exist." The pixel mask algorithm is inherently a tensor broadcast — never implement it as a nested Python loop over (y, x) coordinates.

---

## Common Pitfalls

### Pitfall 1: Subpixel Stroke Degeneration at 64×64

**What goes wrong:** A stroke with `w < 2/(IMG_SIZE-1)` (approximately `w < 0.032` at 64×64) covers zero pixels. The mask is all-False. `draw()` returns the canvas unchanged with no error.

**Why it happens:** `torch.linspace(0, 1, 64)` produces pixel centers with spacing `1/63 ≈ 0.01587`. A half-width smaller than half this spacing never reaches any pixel center. This is correct behavior for a hard rasterizer but surprising if callers expect all strokes to be visible.

**How to avoid:** The neural renderer pre-trainer (Phase 2) must bias 20% of training batches toward minimum-visible params. Document the minimum visible width as `MIN_VISIBLE_W = 2.0 / (IMG_SIZE - 1)` in `config.py` or in a comment in `renderer.py`. Tests for "thin strokes" should use `w = MIN_VISIBLE_W` not `w = 0.001`.

**Warning signs:** Phase 2 renderer produces blank output for small strokes — check that training data includes strokes with `w >= MIN_VISIBLE_W`.

### Pitfall 2: `torch.meshgrid` Deprecation Warning Without `indexing`

**What goes wrong:** Calling `torch.meshgrid(ys, xs)` (without `indexing=`) emits `UserWarning` in PyTorch 2.10 and may silently change behavior in a future release. CI output becomes noisy.

**Why it happens:** PyTorch is transitioning the default from `'ij'` to `'xy'`. Both exist for backward compatibility but the warning appears on every call.

**How to avoid:** Always call `torch.meshgrid(ys, xs, indexing='ij')`. [VERIFIED: in-session test]

### Pitfall 3: Negative LMS Values in oklab Conversion

**What goes wrong:** The LMS intermediate values (`l`, `m`, `s`) can be very slightly negative due to float32 accumulation error, especially near pure black. Calling `.pow(1/3)` on a negative value produces NaN.

**Why it happens:** The M1 matrix has negative off-diagonal entries. For colors very close to (0,0,0), rounding pushes one LMS channel below zero by ~1e-7.

**How to avoid:** Add `.clamp(min=0.0)` before `.pow(1/3)`. This is included in the reference implementation above. [VERIFIED: in-session test]

### Pitfall 4: `torch.cdist` Input Shape Requirements

**What goes wrong:** `torch.cdist` requires at least 2D inputs. Calling it with 1D tensors raises a RuntimeError.

**Why it happens:** `cdist` is designed for batched pairwise distance, not scalar-to-scalar.

**How to avoid:** Always ensure inputs are `(N, D)` shaped. In `project_color`, use `.unsqueeze(0)` to make the query `(1, 3)`. The palette is already `(P, 3)`.

### Pitfall 5: Python Version Mismatch (3.14 vs CLAUDE.md 3.11 Recommendation)

**What goes wrong:** CLAUDE.md recommends Python 3.11 but the installed environment uses Python 3.14.3. PyTorch 2.10.0 works correctly on 3.14 (verified in-session), but future phases that add dependencies may hit packages not yet supporting 3.14.

**Why it happens:** CLAUDE.md was written against pytorch.org's recommended install path. The actual machine has a newer Python.

**How to avoid:** Phase 1 is pure PyTorch + stdlib math, no third-party packages that might lag. Note this as an environment fact. If future packages fail on 3.14, creating a conda environment with 3.11 is the fallback.

### Pitfall 6: Circular Import Between `palette.py` and `config.py`

**What goes wrong:** If `config.py` imports from `palette.py` (e.g., to derive a color count), and `palette.py` imports from `config.py`, Python raises `ImportError: cannot import name`.

**Why it happens:** Circular imports are silent until the import system hits the cycle.

**How to avoid:** `config.py` imports nothing. `palette.py` imports `config.py`. `renderer.py` imports `config.py`. No module imports `palette.py` or `renderer.py` at module level in Phase 1. The dependency graph is a DAG: `config <- palette`, `config <- renderer`.

---

## Code Examples

Verified patterns from in-session testing and official sources:

### Minimal config.py

```python
# Source: Phase 1 design decision table (paint_ai_design.md + CONTEXT.md D-01..D-08)
# No imports. No logic. Constants only.

IMG_SIZE: int = 64
STROKE_DIM: int = 8           # (cx, cy, w, h, theta, r, g, b)
STROKES_PER_STEP: int = 5
N_STROKES: int = 40           # steps per episode; 40 steps x 5 strokes = 200 total
IMAGE_RANGE: tuple = (0.0, 1.0)

# Default colorspace for palette projection
PALETTE_COLORSPACE: str = "rgb"   # options: "rgb", "oklab", "hsv"

# Derived constant (documentation only; not logic)
# MIN_VISIBLE_STROKE_WIDTH = 2.0 / (IMG_SIZE - 1)  # ~0.032 at 64x64
```

### Palette Module Skeleton

```python
# Source: CONTEXT.md D-05, D-06 + in-session verification (2026-06-08)
import torch
from config import PALETTE_COLORSPACE

# Physical paint palette — edit these values after measuring with paint mixer.
# Values are sRGB float in [0.0, 1.0]: divide uint8 (0-255) values by 255.
# Placeholder — user must replace with actual ~40 colors from paint mixer.
_PALETTE_SRGB: list[tuple[float, float, float]] = [
    (1.000, 1.000, 1.000),  # white
    (0.000, 0.000, 0.000),  # black
    # ... ~38 more colors from physical paint mixer
]

# Tensor form — created once at module load, shape (P, 3)
PALETTE: torch.Tensor = torch.tensor(_PALETTE_SRGB, dtype=torch.float32)

def project_color(
    rgb: tuple | list | torch.Tensor,
    colorspace: str = PALETTE_COLORSPACE,
) -> torch.Tensor:
    """Return nearest palette color in sRGB [0,1] using specified colorspace distance."""
    ...  # see full implementation in Code Examples section above
```

### renderer.py Skeleton

```python
# Source: CONTEXT.md D-02..D-04 + in-session verification (2026-06-08)
import torch
import math
from config import IMG_SIZE

@torch.no_grad()
def draw(canvas: torch.Tensor, stroke_params: torch.Tensor) -> torch.Tensor:
    """
    Render an opaque oriented rectangle onto canvas.

    canvas:        float32 (3, H, W) in [0, 1]
    stroke_params: float32 (8,) = (cx, cy, w, h, theta_01, r, g, b), all in [0, 1]
                   theta_01 ∈ [0,1] maps to [0, π]

    Returns new canvas tensor. No autograd graph attached.
    """
    ...  # see full implementation in Architecture Patterns section above
```

### Pytest test skeleton for renderer

```python
# Source: pytest docs + in-session design (2026-06-08)
import torch
import pytest
from renderer import draw

def test_draw_output_shape():
    canvas = torch.zeros(3, 64, 64)
    params = torch.tensor([0.5, 0.5, 0.3, 0.1, 0.0, 1.0, 0.0, 0.0])
    result = draw(canvas, params)
    assert result.shape == (3, 64, 64)

def test_draw_no_autograd():
    canvas = torch.zeros(3, 64, 64)
    params = torch.tensor([0.5, 0.5, 0.3, 0.1, 0.0, 1.0, 0.0, 0.0])
    result = draw(canvas, params)
    assert not result.requires_grad

def test_draw_paints_pixels():
    canvas = torch.zeros(3, 64, 64)
    params = torch.tensor([0.5, 0.5, 0.3, 0.2, 0.0, 1.0, 0.0, 0.0])
    result = draw(canvas, params)
    assert (result > 0).any()

def test_draw_full_canvas():
    canvas = torch.zeros(3, 64, 64)
    params = torch.tensor([0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    result = draw(canvas, params)
    assert (result[1] > 0).sum().item() == 64 * 64

def test_draw_subpixel_stroke_is_empty():
    """Stroke smaller than a pixel is correct (returns unmodified canvas)."""
    canvas = torch.zeros(3, 64, 64)
    params = torch.tensor([0.5, 0.5, 0.001, 0.001, 0.0, 1.0, 0.0, 0.0])
    result = draw(canvas, params)
    assert torch.equal(result, canvas)

def test_draw_values_in_range():
    canvas = torch.zeros(3, 64, 64)
    params = torch.tensor([0.3, 0.7, 0.2, 0.1, 0.25, 0.8, 0.5, 0.2])
    result = draw(canvas, params)
    assert result.min() >= 0.0 and result.max() <= 1.0

def test_draw_gpu():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    device = torch.device("cuda")
    canvas = torch.zeros(3, 64, 64, device=device)
    params = torch.tensor([0.5, 0.5, 0.3, 0.2, 0.0, 1.0, 0.0, 0.0], device=device)
    result = draw(canvas, params)
    assert result.device.type == "cuda"
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.meshgrid(y, x)` (implicit 'ij') | `torch.meshgrid(y, x, indexing='ij')` | PyTorch 1.10 (deprecation); future release will break | Always pass `indexing='ij'` |
| Manual LMS computation for perceptual color | okLab (Björn Ottosson, 2020) | 2020 — now integrated in CSS Color Level 5, Photoshop, Godot, Unity | okLab is the modern standard for perceptual color distance |
| sRGB Euclidean distance for palette search | okLab L2 for palette search | Gradual adoption 2020-2024 | More visually accurate nearest-color; CLAUDE.md decision D-06 supports both |

**Deprecated/outdated:**
- `torch.meshgrid` without `indexing=`: Will become an error in a future PyTorch version. Never omit `indexing`.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `math.cos(theta_val)` (Python scalar, not tensor) is the cleanest approach when `theta_01` is a 0-dim tensor extracted via `.item()` | Architecture Patterns | If stroke_params is always a Python list (not tensor), `.item()` is unnecessary — but harmless |
| A2 | The physical paint palette will have ~40 colors and all will fit comfortably in a `torch.tensor` literal in `palette.py` | Code Examples | If palette is 400+ colors, a CSV-loaded approach is better — but CLAUDE.md says ~40 |
| A3 | Python 3.14 will not cause issues in Phase 1 through Phase 5 for the listed dependencies (torch, numpy, matplotlib, imageio, tqdm) | Environment Availability | imageio and tensorboard haven't been tested on 3.14 yet; may need conda env with 3.11 if they fail |

**A1 and A2 are low-risk assumptions for Phase 1 specifically.**

---

## Open Questions

1. **Palette color entry format**
   - What we know: CONTEXT.md D-05 says "divide paint mixer's 0–255 values by 255 before entering"
   - What's unclear: The user has ~40 actual physical paint colors but they are not yet entered in code. Phase 1 will include a placeholder palette; the actual colors will be filled in by the user.
   - Recommendation: Ship `palette.py` with a clearly marked placeholder list of 5-6 colors and a comment instructing the user to replace it. The `project_color` function doesn't care about the palette size.

2. **Should `config.py` export `MIN_VISIBLE_STROKE_WIDTH`?**
   - What we know: At 64×64, any stroke with `w < 2/(63)` ≈ 0.032 is invisible
   - What's unclear: This is a property of the resolution, not a hyperparameter. It changes when IMG_SIZE changes.
   - Recommendation: Add as a comment in `renderer.py`, not as a constant in `config.py`. It's derivable and not configurable.

3. **Default colorspace for `project_color`**
   - What we know: CONTEXT.md D-06 says "configurable in `config.py`" with no stated default
   - What's unclear: Which colorspace should be the default?
   - Recommendation: Default to `"rgb"` (Euclidean L2 in sRGB). It's the simplest baseline. okLab is available for eval-time experimentation without touching training code.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | All | ✓ | 3.14.3 (installed) | — |
| PyTorch | renderer.py, palette.py | ✓ | 2.10.0+cu126 | — |
| CUDA | GPU execution | ✓ | CUDA 12.6 (GTX 1660 Ti) | CPU fallback (slower) |
| pytest | Tests | ✓ | 8.4.2 | — |
| numpy | Indirect deps | ✓ | 2.3.5 | — |
| torchvision | Phase 2+ (save_image) | ✓ | 0.25.0+cu126 | — |
| matplotlib | Phase 2+ | ✓ | 3.10.8 | — |
| tqdm | Phase 2+ | ✓ | 4.67.3 | — |
| tensorboard | Phase 4 | ✗ | — | Install: `pip install tensorboard` |
| imageio | Phase 5 | ✗ | — | Install: `pip install imageio imageio-ffmpeg` |
| opencv-python | Optional (hard rasterizer alt) | ✗ | — | Not needed — pure PyTorch rasterizer chosen per D-02 |

**Missing dependencies with no fallback:** None for Phase 1.

**Missing dependencies with fallback:** tensorboard (Phase 4) and imageio (Phase 5) need installation. Neither is required for Phase 1.

**Python version note:** CLAUDE.md recommends Python 3.11 but the installed environment uses 3.14.3. PyTorch 2.10.0+cu126 is verified working on Python 3.14 for Phase 1's requirements (meshgrid, cdist, where, no_grad). If future phases hit Python 3.14 incompatibilities with other packages, create a conda environment: `conda create -n paint_ai python=3.11`.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.4.2 (installed) |
| Config file | `pyproject.toml` (to be created in Wave 0) |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FOUND-01 | `config.py` exports correct constants with correct types | unit | `pytest tests/test_config.py -x` | ❌ Wave 0 |
| FOUND-02 | `project_color` returns nearest palette color in all 3 colorspaces | unit | `pytest tests/test_palette.py -x` | ❌ Wave 0 |
| FOUND-02 | `project_color` handles batch input (if supported) | unit | `pytest tests/test_palette.py -x` | ❌ Wave 0 |
| FOUND-03 | `draw()` output shape `(3, H, W)` | unit | `pytest tests/test_renderer.py::test_draw_output_shape -x` | ❌ Wave 0 |
| FOUND-03 | `draw()` no autograd on output | unit | `pytest tests/test_renderer.py::test_draw_no_autograd -x` | ❌ Wave 0 |
| FOUND-03 | `draw()` paints visible pixels for normal strokes | unit | `pytest tests/test_renderer.py::test_draw_paints_pixels -x` | ❌ Wave 0 |
| FOUND-03 | `draw()` handles full-canvas stroke (`w=h=1`) | unit | `pytest tests/test_renderer.py::test_draw_full_canvas -x` | ❌ Wave 0 |
| FOUND-03 | `draw()` handles subpixel stroke (w<0.02, correct empty behavior) | unit | `pytest tests/test_renderer.py::test_draw_subpixel_stroke_is_empty -x` | ❌ Wave 0 |
| FOUND-03 | `draw()` handles extreme rotations (θ≈0, θ≈0.5, θ≈1.0) | unit | `pytest tests/test_renderer.py::test_draw_extreme_rotations -x` | ❌ Wave 0 |
| FOUND-03 | `draw()` works on CUDA device | unit | `pytest tests/test_renderer.py::test_draw_gpu -x` | ❌ Wave 0 |
| FOUND-01..03 | All modules importable without circular deps | unit | `pytest tests/test_imports.py -x` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `pytest tests/ -x -q`
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/__init__.py` — empty, marks tests as package
- [ ] `tests/test_config.py` — covers FOUND-01
- [ ] `tests/test_palette.py` — covers FOUND-02
- [ ] `tests/test_renderer.py` — covers FOUND-03
- [ ] `tests/test_imports.py` — circular dependency check
- [ ] `pyproject.toml` — pytest config with `testpaths = ["tests"]`
- [ ] `models/__init__.py` — empty, needed for future phases
- [ ] `ddpg/__init__.py` — empty, needed for future phases

---

## Security Domain

> `security_enforcement: true` per config.json, ASVS Level 1.

Phase 1 has no authentication, no network I/O, no file uploads, no user-supplied strings, and no database. The security surface is minimal.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | No auth in Phase 1 |
| V3 Session Management | No | No sessions |
| V4 Access Control | No | No access control |
| V5 Input Validation | Partial | Stroke params are floats in [0,1]; clamp inputs rather than raise |
| V6 Cryptography | No | No secrets or encryption |
| V7 Error Handling | Yes | Don't expose raw tracebacks in production; use assertions in tests |

### Known Threat Patterns for PyTorch ML projects

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| NaN propagation from invalid float ops | Tampering | `clamp(min=0)` before `pow(1/3)` in oklab; assert output is finite in tests |
| Untrusted stroke params causing OOB tensor access | Tampering | PyTorch tensor ops are bounds-safe; no raw memory access; no special mitigation needed |
| Model weight file (Phase 2+) loading arbitrary code via pickle | Tampering | Use `torch.load(..., weights_only=True)` in Phase 2+; not relevant for Phase 1 |

**Phase 1 security posture:** Acceptable for local development. No user input, no file I/O beyond `config.py` constants. No special mitigations required beyond the NaN guard in oklab conversion.

---

## Sources

### Primary (HIGH confidence)

- In-session Python execution — all draw(), project_color(), colorspace math verified in PyTorch 2.10.0+cu126, Python 3.14 environment
- [bottosson.github.io/posts/oklab/](https://bottosson.github.io/posts/oklab/) — Exact oklab conversion matrices (canonical source, Björn Ottosson)
- CONTEXT.md D-01..D-08 — Locked design decisions
- paint_ai_design.md — Module API contracts and architectural rationale
- REQUIREMENTS.md — FOUND-01, FOUND-02, FOUND-03 acceptance criteria

### Secondary (MEDIUM confidence)

- [docs.pytorch.org/docs/2.12/generated/torch.meshgrid.html](https://docs.pytorch.org/docs/2.12/generated/torch.meshgrid.html) — `indexing='ij'` required, deprecation confirmed
- [docs.pytorch.org/docs/2.12/generated/torch.cdist.html](https://docs.pytorch.org/docs/2.12/generated/torch.cdist.html) — `(N, D)` input shape confirmed, `p=2` is L2
- [docs.pytorch.org/docs/2.12/generated/torch.no_grad.html](https://docs.pytorch.org/docs/2.12/generated/torch.no_grad.html) — Decorator usage confirmed

### Tertiary (LOW confidence)

- None. All claims are verified or cited.

---

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — all packages verified installed and functional in-session
- Rasterizer algorithm: HIGH — rotation matrix + meshgrid pixel test verified working on GPU with multiple edge cases
- okLab conversion: HIGH — formulas fetched from canonical source and verified numerically in-session
- HSV conversion: HIGH — standard formula, verified in-session
- `torch.cdist` for palette NN: HIGH — API confirmed from docs, tested in-session
- Pitfalls: HIGH — all confirmed via in-session code execution

**Research date:** 2026-06-08
**Valid until:** 2026-08-01 (stable libraries; PyTorch API very unlikely to change for these primitives)
