---
phase: 01-foundation
reviewed: 2026-06-09T00:00:00Z
depth: standard
files_reviewed: 11
files_reviewed_list:
  - config.py
  - palette.py
  - renderer.py
  - pyproject.toml
  - tests/test_config.py
  - tests/test_palette.py
  - tests/test_renderer.py
  - tests/test_imports.py
  - models/__init__.py
  - ddpg/__init__.py
  - tests/__init__.py
findings:
  critical: 2
  warning: 3
  info: 3
  total: 8
status: issues_found
---

# Phase 01: Code Review Report

**Reviewed:** 2026-06-09
**Depth:** standard
**Files Reviewed:** 11
**Status:** issues_found

## Summary

The foundation phase delivers `config.py` (constants only), `palette.py` (colorspace-aware nearest-neighbor projection), and `renderer.py` (hard rasterizer for oriented rectangles). The `models/`, `ddpg/`, and `tests/` package stubs are empty. Core logic is concise and the math is correct for the nominal case.

Two critical defects were confirmed by execution:

1. `project_color` returns a live view into the module-level `PALETTE` tensor. Any caller that performs an in-place operation on the returned value silently corrupts the global palette for the rest of the process — verified by running the mutation test.
2. The HSV colorspace option misclassifies colors whose hue wraps near the red/magenta boundary (H ≈ 0 ≡ H ≈ 1). The Euclidean L2 distance used by `torch.cdist` treats hue as a linear axis, so a color at H=0.99 (visually almost pure red) is assigned to blue (H≈0.67) rather than red (H≈0.0) — verified empirically.

Three warnings are also present: an unused import in `renderer.py`, absent device-consistency enforcement in both `draw()` and `project_color()`, and no output clamping in `draw()`.

## Critical Issues

### CR-01: `project_color` returns a mutable view of `PALETTE` — silent global state corruption

**File:** `palette.py:101`
**Issue:** `return PALETTE[idx]` returns a row of the module-level `PALETTE` tensor by reference (tensor slicing in PyTorch returns a view, not a copy). Any caller that performs an in-place operation on the returned value (`result[0] = ...`, `result.fill_(...)`, `result.clamp_(...)`, etc.) directly mutates the shared palette. This was confirmed by executing the mutation:

```python
result = project_color((1.0, 0.0, 0.0))
result[0] = 0.5          # in-place write
# PALETTE[2] is now [0.5, 0.0, 0.0] — permanently corrupted for this process
```

In the DDPG training loop, palette-projected colors will be placed directly onto tensors and potentially passed through in-place normalization or augmentation steps, making this a realistic corruption path.

**Fix:** Return a copy so the caller owns the tensor:
```python
# palette.py line 101
return PALETTE[idx].clone()
```

---

### CR-02: HSV colorspace projection has hue wraparound bug — misclassifies colors near H=0/H=1 boundary

**File:** `palette.py:58–61` (hue computation in `_rgb_to_hsv`) and `palette.py:99` (cdist call)
**Issue:** Hue in HSV is a circular quantity: H=0 and H=1 both represent red. `torch.cdist` computes Euclidean L2 distance treating H as a linear value in [0, 1). A query color at H=0.99 (a very slightly blue-tinted red, perceptually almost identical to pure red) has:
- Distance to red palette entry (H=0.0): **0.99**
- Distance to blue palette entry (H=0.667): **0.323**

Result: the color is projected to blue, not red. Verified empirically:

```python
project_color([1.0, 0.0, 0.05], colorspace="hsv")
# Returns [0., 0., 1.]  (blue)
# Correct answer: [1., 0., 0.]  (red)
```

This is a correctness failure for any use of `colorspace="hsv"`. The default is `"rgb"` (`PALETTE_COLORSPACE = "rgb"` in `config.py`), so this bug is dormant unless a user explicitly switches to HSV. However, the option is documented and tested, so it is expected to work correctly.

**Fix — Option A (simplest):** Remove the HSV colorspace option entirely if there is no current use case for it. Keep only `"rgb"` and `"oklab"`.

**Fix — Option B:** Wrap the hue distance with a circular correction before calling cdist:
```python
# palette.py — new helper
def _hsv_circular_dist(q_hsv: torch.Tensor, pal_hsv: torch.Tensor) -> torch.Tensor:
    """L2 distance in HSV with circular hue correction."""
    dh = (q_hsv[..., 0:1] - pal_hsv[..., 0:1]).abs()
    dh = torch.minimum(dh, 1.0 - dh)          # wrap: max circular dist is 0.5
    dsv = q_hsv[..., 1:] - pal_hsv[..., 1:]   # (1, P, 2)
    # broadcast and combine
    d = torch.cat([dh.unsqueeze(-1) if needed..., dsv], dim=-1)
    return d.pow(2).sum(-1).sqrt()             # (1, P)
```
Then replace the `torch.cdist` call with this function when `colorspace == "hsv"`.

---

## Warnings

### WR-01: `IMG_SIZE` imported in `renderer.py` but never used — dead import

**File:** `renderer.py:5`
**Issue:** `from config import IMG_SIZE` is present but `IMG_SIZE` is never referenced in `renderer.py`. The function derives canvas dimensions from `canvas.shape[-2]` and `canvas.shape[-1]` at runtime (correct approach). The dead import adds misleading coupling that implies `draw()` is constrained to `IMG_SIZE × IMG_SIZE` canvases when it is not.

**Fix:**
```python
# Remove line 5:
# from config import IMG_SIZE
```

---

### WR-02: `draw()` has no device-consistency enforcement — silent RuntimeError in GPU training

**File:** `renderer.py:11–52`
**Issue:** `draw()` creates the pixel grid with `device=canvas.device` (line 32–33) but then does arithmetic between that grid and `cx`, `cy` unpacked from `stroke_params` (line 37–38). If `canvas` is on CUDA and `stroke_params` is on CPU (a common mistake when manually constructing test params), the subtraction `grid_x - cx` raises an unguarded `RuntimeError: Expected all tensors to be on the same device`. The error message from PyTorch does not point to `draw()` clearly. The same issue applies to `torch.stack([r, g, b]).expand_as(canvas)` at line 51.

The GPU test in `test_renderer.py` (line 59–66) creates BOTH tensors on CUDA, so it does not cover the mismatch scenario.

**Fix:** Add an assertion at the top of `draw()`:
```python
def draw(canvas: torch.Tensor, stroke_params: torch.Tensor) -> torch.Tensor:
    assert canvas.device == stroke_params.device, (
        f"draw(): canvas ({canvas.device}) and stroke_params ({stroke_params.device}) "
        "must be on the same device"
    )
    ...
```

---

### WR-03: `draw()` does not clamp output — out-of-range stroke params silently produce invalid canvas values

**File:** `renderer.py:52`
**Issue:** The docstring states all `stroke_params` values are in `[0, 1]`, but `draw()` does not enforce this. If `r`, `g`, or `b` are outside `[0, 1]` (e.g., from an early-stage untrained actor), `torch.where(mask, color, canvas)` writes those out-of-range values directly into the returned canvas. Downstream L2 loss computation against a `[0, 1]` target image then produces gradient signals that are not representative of actual painting quality.

Verified: passing `r=2.0, b=-1.0` results in `canvas.min() == -1.0` and `canvas.max() == 2.0`.

**Fix:** Clamp the color before compositing:
```python
# renderer.py line 51
color = torch.stack([r, g, b]).clamp(0.0, 1.0).view(3, 1, 1).expand_as(canvas)
```

---

## Info

### IN-01: `config.py` — `IMAGE_RANGE` type annotation loses precision

**File:** `config.py:7`
**Issue:** `IMAGE_RANGE: tuple = (0.0, 1.0)` uses the bare `tuple` annotation. Python 3.9+ supports `tuple[float, float]`, which is more precise and makes it clear to type checkers that this is a 2-element float tuple.

**Fix:**
```python
IMAGE_RANGE: tuple[float, float] = (0.0, 1.0)
```

---

### IN-02: `test_project_color_invalid_colorspace` catches too many exception types

**File:** `tests/test_palette.py:35`
**Issue:** `pytest.raises((ValueError, KeyError, AssertionError))` accepts three distinct exception types, but `project_color` only raises `ValueError`. Catching `KeyError` and `AssertionError` weakens the test contract — a future refactor that changes the exception type would silently pass.

**Fix:**
```python
def test_project_color_invalid_colorspace():
    with pytest.raises(ValueError, match="Unsupported colorspace"):
        project_color((0.5, 0.5, 0.5), colorspace="xyz")
```

---

### IN-03: `test_renderer.py` GPU test does not cover device-mismatch scenario

**File:** `tests/test_renderer.py:59–66`
**Issue:** `test_draw_gpu` places both `canvas` and `params` on CUDA. The real-world bug (CPU params + GPU canvas) is not exercised. This means WR-02 is not caught by the existing test suite.

**Fix:** Add a second GPU test:
```python
def test_draw_gpu_device_mismatch():
    """Mismatched devices must raise AssertionError, not an opaque RuntimeError."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    canvas = torch.zeros(3, 64, 64, device="cuda")
    params = torch.tensor([0.5, 0.5, 0.3, 0.2, 0.0, 1.0, 0.0, 0.0])  # CPU
    with pytest.raises(AssertionError, match="same device"):
        draw(canvas, params)
```

---

_Reviewed: 2026-06-09_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
