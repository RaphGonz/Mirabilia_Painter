# Phase 1: Foundation - Pattern Map

**Mapped:** 2026-06-08
**Files analyzed:** 8 (3 source modules + 4 test files + 1 config file)
**Analogs found:** 0 / 8 — greenfield project, no existing Python source

---

## File Classification

| New File | Role | Data Flow | Closest Analog | Match Quality |
|----------|------|-----------|----------------|---------------|
| `config.py` | config | — | No analog — greenfield | N/A |
| `palette.py` | utility | transform | No analog — greenfield | N/A |
| `renderer.py` | utility | transform | No analog — greenfield | N/A |
| `tests/test_config.py` | test | — | No analog — greenfield | N/A |
| `tests/test_palette.py` | test | — | No analog — greenfield | N/A |
| `tests/test_renderer.py` | test | — | No analog — greenfield | N/A |
| `tests/test_imports.py` | test | — | No analog — greenfield | N/A |
| `pyproject.toml` | config | — | No analog — greenfield | N/A |

**Supporting scaffold (empty init files):** `models/__init__.py`, `ddpg/__init__.py`, `tests/__init__.py`

---

## Pattern Assignments

All patterns below are sourced from RESEARCH.md in-session verified code (2026-06-08). No codebase analogs exist. Planner must use these excerpts directly.

---

### `config.py` (config, no data flow)

**Source:** RESEARCH.md "Minimal config.py" + decisions D-01..D-08

**Full file pattern:**
```python
# No imports. No logic. Constants only.

IMG_SIZE: int = 64
STROKE_DIM: int = 8           # (cx, cy, w, h, theta, r, g, b)
STROKES_PER_STEP: int = 5
N_STROKES: int = 40           # steps per episode; 40 steps x 5 strokes = 200 total
IMAGE_RANGE: tuple = (0.0, 1.0)

# Default colorspace for palette projection
PALETTE_COLORSPACE: str = "rgb"   # options: "rgb", "oklab", "hsv"

# MIN_VISIBLE_STROKE_WIDTH = 2.0 / (IMG_SIZE - 1)  # ~0.032 at 64x64 (comment only, not a constant)
```

**Critical constraints:**
- Zero imports — not even stdlib
- Zero logic — no `if`, no function calls, no derived constants computed at module load
- `PALETTE_COLORSPACE` default is `"rgb"` per open question resolution in RESEARCH.md

---

### `palette.py` (utility, transform)

**Source:** RESEARCH.md "Pattern 2: Colorspace-Aware Nearest-Neighbor Projection" + "Palette Module Skeleton"

**Imports pattern:**
```python
import torch
import math
from config import PALETTE_COLORSPACE
```

**Palette storage pattern — create tensor once at module load:**
```python
_PALETTE_SRGB: list[tuple[float, float, float]] = [
    (1.000, 1.000, 1.000),  # white
    (0.000, 0.000, 0.000),  # black
    # ... ~38 more colors from physical paint mixer
    # User fills in: divide uint8 values by 255.0
]

# Shape (P, 3), float32 — created once, not per-call
PALETTE: torch.Tensor = torch.tensor(_PALETTE_SRGB, dtype=torch.float32)
```

**okLab conversion pattern (sRGB linearize → M1 → cbrt → M2):**
```python
def _srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    """Apply inverse sRGB gamma. [VERIFIED: sRGB IEC 61966-2-1 spec]"""
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def _linear_to_oklab(rgb_linear: torch.Tensor) -> torch.Tensor:
    """Convert linear sRGB to oklab. rgb_linear: (..., 3). Returns (..., 3) [L, a, b].
    Source: bottosson.github.io/posts/oklab/"""
    r, g, b = rgb_linear[..., 0], rgb_linear[..., 1], rgb_linear[..., 2]
    l = 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
    m = 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
    s = 0.0883024619*r + 0.2817188376*g + 0.6299787005*b
    # clamp before pow(1/3) — LMS can be tiny-negative due to float32 error near black
    l_ = l.clamp(min=0.0).pow(1.0 / 3.0)
    m_ = m.clamp(min=0.0).pow(1.0 / 3.0)
    s_ = s.clamp(min=0.0).pow(1.0 / 3.0)
    L  =  0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_
    a  =  1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_
    b_ =  0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_
    return torch.stack([L, a, b_], dim=-1)
```

**HSV conversion pattern (branch-free with torch.where):**
```python
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
```

**Core project_color pattern (cdist nearest-neighbor):**
```python
def project_color(
    rgb: tuple | list | torch.Tensor,
    colorspace: str = PALETTE_COLORSPACE,
) -> torch.Tensor:
    """Return nearest palette color in sRGB [0,1] using specified colorspace distance."""
    q = torch.as_tensor(rgb, dtype=torch.float32).unsqueeze(0)  # (1, 3) — cdist requires 2D
    pal = PALETTE.float()

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
    return PALETTE[idx]
```

**Anti-patterns to avoid in this file:**
- Do NOT convert `_PALETTE_SRGB` to tensor on every call — do it once at module load
- Do NOT call `torch.cdist` on 1D inputs — always `.unsqueeze(0)` the query to make `(1, 3)`
- Do NOT use a Python loop over palette entries — use `torch.cdist` vectorized

---

### `renderer.py` (utility, transform)

**Source:** RESEARCH.md "Pattern 1: Rotation-Matrix Pixel Mask" + "renderer.py Skeleton"

**Imports pattern:**
```python
import torch
import math
from config import IMG_SIZE
```

**Core draw() pattern (rotation matrix + meshgrid pixel mask):**
```python
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
    theta = theta_01.item() * math.pi   # scalar via .item() — avoids device mismatch

    H_px, W_px = canvas.shape[-2], canvas.shape[-1]
    device = canvas.device

    # Pixel coordinate grid in [0, 1] x [0, 1]
    ys = torch.linspace(0.0, 1.0, H_px, device=device)
    xs = torch.linspace(0.0, 1.0, W_px, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # always pass indexing='ij'

    # Translate to rectangle-center-relative coordinates
    dx = grid_x - cx
    dy = grid_y - cy

    # Rotate pixel offsets into rectangle's local frame
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    local_x =  cos_t * dx + sin_t * dy
    local_y = -sin_t * dx + cos_t * dy

    # Boolean mask: pixel inside rectangle
    mask = (local_x.abs() <= w / 2.0) & (local_y.abs() <= h / 2.0)  # (H, W) bool
    mask = mask.unsqueeze(0)  # (1, H, W) broadcast with (3, H, W)

    # Paint with mask
    color = torch.stack([r, g, b]).view(3, 1, 1).expand_as(canvas)
    return torch.where(mask, color, canvas)
```

**Critical constraints:**
- `@torch.no_grad()` decorator is mandatory — downstream usage (pretrain_renderer.py Phase 2) depends on no autograd graph
- Always `indexing='ij'` in `torch.meshgrid` — without it: UserWarning + future breakage
- Extract scalar with `.item()` before `math.cos/sin` — calling `math.cos` on a tensor raises TypeError
- Subpixel strokes (w < ~0.032 at 64x64) return unmodified canvas — this is correct behavior, not a bug

---

### `tests/test_config.py` (test)

**Source:** RESEARCH.md "Validation Architecture" — FOUND-01 test map

**Pattern:**
```python
import pytest
from config import IMG_SIZE, STROKE_DIM, STROKES_PER_STEP, N_STROKES, IMAGE_RANGE, PALETTE_COLORSPACE

def test_img_size():
    assert IMG_SIZE == 64

def test_stroke_dim():
    assert STROKE_DIM == 8

def test_strokes_per_step():
    assert STROKES_PER_STEP == 5

def test_n_strokes():
    assert N_STROKES == 40

def test_image_range():
    assert IMAGE_RANGE == (0.0, 1.0)

def test_palette_colorspace_valid():
    assert PALETTE_COLORSPACE in {"rgb", "oklab", "hsv"}

def test_all_constants_types():
    assert isinstance(IMG_SIZE, int)
    assert isinstance(STROKE_DIM, int)
    assert isinstance(STROKES_PER_STEP, int)
    assert isinstance(N_STROKES, int)
    assert isinstance(IMAGE_RANGE, tuple)
    assert isinstance(PALETTE_COLORSPACE, str)
```

---

### `tests/test_renderer.py` (test)

**Source:** RESEARCH.md "Pytest test skeleton for renderer" + FOUND-03 test map

**Pattern (copy entire block):**
```python
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
    """Stroke smaller than a pixel — correct behavior is unmodified canvas."""
    canvas = torch.zeros(3, 64, 64)
    params = torch.tensor([0.5, 0.5, 0.001, 0.001, 0.0, 1.0, 0.0, 0.0])
    result = draw(canvas, params)
    assert torch.equal(result, canvas)

def test_draw_values_in_range():
    canvas = torch.zeros(3, 64, 64)
    params = torch.tensor([0.3, 0.7, 0.2, 0.1, 0.25, 0.8, 0.5, 0.2])
    result = draw(canvas, params)
    assert result.min() >= 0.0 and result.max() <= 1.0

def test_draw_extreme_rotations():
    """Verify draw works at theta boundaries: 0.0, 0.5 (90 deg), 1.0 (180 deg)."""
    canvas = torch.zeros(3, 64, 64)
    for theta in [0.0, 0.5, 1.0]:
        params = torch.tensor([0.5, 0.5, 0.3, 0.1, theta, 1.0, 0.0, 0.0])
        result = draw(canvas, params)
        assert result.shape == (3, 64, 64)
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

### `tests/test_palette.py` (test)

**Source:** RESEARCH.md FOUND-02 test map + project_color API

**Pattern:**
```python
import torch
import pytest
from palette import project_color, PALETTE

def test_project_color_returns_tensor():
    result = project_color((1.0, 0.0, 0.0))
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3,)

def test_project_color_in_palette():
    """Result must be one of the palette entries."""
    result = project_color((0.5, 0.5, 0.5))
    matches = (PALETTE == result.unsqueeze(0)).all(dim=1)
    assert matches.any()

def test_project_color_rgb():
    result = project_color((1.0, 1.0, 1.0), colorspace="rgb")
    assert result.shape == (3,)

def test_project_color_oklab():
    result = project_color((1.0, 1.0, 1.0), colorspace="oklab")
    assert result.shape == (3,)

def test_project_color_hsv():
    result = project_color((1.0, 1.0, 1.0), colorspace="hsv")
    assert result.shape == (3,)

def test_project_color_invalid_colorspace():
    with pytest.raises((ValueError, KeyError, AssertionError)):
        project_color((0.5, 0.5, 0.5), colorspace="xyz")

def test_project_color_black():
    """Black input — oklab NaN guard must hold."""
    result = project_color((0.0, 0.0, 0.0), colorspace="oklab")
    assert result.shape == (3,)
    assert not result.isnan().any()

def test_project_color_accepts_tensor():
    rgb = torch.tensor([0.5, 0.3, 0.1])
    result = project_color(rgb)
    assert result.shape == (3,)
```

---

### `tests/test_imports.py` (test)

**Source:** RESEARCH.md circular dependency DAG — `config <- palette`, `config <- renderer`

**Pattern:**
```python
def test_config_importable():
    import config  # noqa: F401

def test_palette_importable():
    import palette  # noqa: F401

def test_renderer_importable():
    import renderer  # noqa: F401

def test_no_circular_import():
    """Import order matches DAG: config has no deps; palette and renderer depend on config only."""
    import config
    import palette
    import renderer
    # If this doesn't raise, no circular import occurred
```

---

### `pyproject.toml` (config)

**Source:** RESEARCH.md "Validation Architecture"

**Pattern:**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

---

## Shared Patterns

### no-logic constants module
**Apply to:** `config.py`

The constant module contains zero imports and zero logic. Any value that requires computation (even `TOTAL_STROKES = N_STROKES * STROKES_PER_STEP`) belongs in the caller, not in config. Violation breaks the guarantee that importing config has no side effects and no failure modes.

### @torch.no_grad() decorator
**Apply to:** `renderer.draw()`

```python
@torch.no_grad()
def draw(...) -> torch.Tensor:
```

Both `@torch.no_grad()` AND never removing `requires_grad` from outputs are required. The decorator alone is sufficient here; `requires_grad_(False)` is only needed when freezing `nn.Module` parameters (Phase 2+).

### torch.cdist shape contract
**Apply to:** `palette.project_color()`

`torch.cdist` requires `(N, D)` shaped inputs. Always `.unsqueeze(0)` any single query vector before passing to `cdist`. Palette tensor is already `(P, 3)` so no change needed there.

### indexing='ij' in torch.meshgrid
**Apply to:** `renderer.draw()`

Always: `torch.meshgrid(ys, xs, indexing='ij')`. Never omit the `indexing` kwarg.

### .item() before math.cos/sin
**Apply to:** `renderer.draw()`

`theta_01` arrives as a 0-dim tensor from `stroke_params` unpacking. Call `.item()` to get a Python float before `math.cos`/`math.sin`. Calling `math.cos(tensor)` raises TypeError.

### clamp before pow(1/3) in oklab
**Apply to:** `palette._linear_to_oklab()`

```python
l_ = l.clamp(min=0.0).pow(1.0 / 3.0)
```

LMS channels can be slightly negative near pure black due to float32 accumulation. Without the clamp, `pow(1/3)` on a negative produces NaN.

### flat absolute imports
**Apply to:** All Phase 1 modules

```python
from config import IMG_SIZE          # correct
# from .config import IMG_SIZE       # WRONG — no relative imports
# import paint_ai.config             # WRONG — no package prefix
```

---

## No Analog Found

All files in Phase 1 have no codebase analog. The project is greenfield.

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `config.py` | config | — | No existing Python source |
| `palette.py` | utility | transform | No existing Python source |
| `renderer.py` | utility | transform | No existing Python source |
| `tests/test_config.py` | test | — | No existing Python source |
| `tests/test_palette.py` | test | — | No existing Python source |
| `tests/test_renderer.py` | test | — | No existing Python source |
| `tests/test_imports.py` | test | — | No existing Python source |
| `pyproject.toml` | config | — | No existing config files |

Planner must use RESEARCH.md patterns (reproduced verbatim above) as the implementation reference.

---

## Metadata

**Analog search scope:** `C:\Users\raphg\Desktop\Mirabilia\episode1\code\` (full project root)
**Python files found:** 0
**Pattern extraction date:** 2026-06-08
**All patterns sourced from:** RESEARCH.md in-session verified code (PyTorch 2.10.0+cu126, Python 3.14.3, GTX 1660 Ti)
