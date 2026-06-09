# Phase 2: Neural Renderer - Pattern Map

**Mapped:** 2026-06-09
**Files analyzed:** 3 new files
**Analogs found:** 3 / 3

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `models/renderer.py` | model (nn.Module) | transform (params → image) | `renderer.py` | role-match (same domain: stroke rendering; different: hard vs. neural) |
| `pretrain_renderer.py` | training script | batch / file-I/O | `renderer.py` + `palette.py` | partial (import style, config usage, `@torch.no_grad()` guard) |
| `tests/test_neural_renderer.py` | test | request-response (unit) | `tests/test_renderer.py` | exact (same pattern: shape, range, edge-case, GPU skip) |

---

## Pattern Assignments

### `models/renderer.py` (model, transform)

**Analog:** `renderer.py` (hard rasterizer)

**Imports pattern** (`renderer.py` lines 1–5):
```python
import torch
import math
from config import IMG_SIZE
```
Apply to `models/renderer.py` — same convention: stdlib first, then torch, then project config import `from config import IMG_SIZE, STROKE_DIM`. Drop `math` (no angle arithmetic in R).

**Module-level docstring / source comment convention** (`renderer.py` line 1):
```python
# Source: CONTEXT.md D-02..D-04 + in-session verification (2026-06-08)
# Hard rasterizer — opaque oriented rectangle via pure PyTorch tensor ops. No autograd. Pure tensor ops only.
```
Copy the source-citation header. For `models/renderer.py` use:
```python
# Source: CONTEXT.md D-09/D-10/D-11 + in-session verification (2026-06-09)
# NeuralRenderer — differentiable CNN decoder. Input (batch, 8) → output (batch, 3, IMG_SIZE, IMG_SIZE).
# No BatchNorm (D-11). No Dropout. Frozen during RL training (see env.py load_frozen_renderer).
```

**Core pattern — nn.Module class structure** (verbatim from RESEARCH.md Pattern 1, verified in-session):
```python
import torch
import torch.nn as nn
from config import IMG_SIZE, STROKE_DIM


class NeuralRenderer(nn.Module):
    """
    Differentiable neural renderer R.

    Maps stroke parameters (batch, STROKE_DIM) to stroke images (batch, 3, IMG_SIZE, IMG_SIZE).
    Pre-trained against the hard rasterizer; frozen during RL training.

    No BatchNorm (interacts poorly with single-sample inference in env.py).
    No Dropout.
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(STROKE_DIM, 512)

        # Stage 1: 2x2 -> 4x4, 128 -> 64 channels
        self.stage1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
        )
        # Stage 2: 4x4 -> 8x8, 64 -> 32 channels
        self.stage2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
        )
        # Stage 3: 8x8 -> 16x16, 32 -> 16 channels
        self.stage3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
        )
        # Stage 4: 16x16 -> 64x64 — scale_factor=4, NOT 2 (see Pitfall 1 in RESEARCH.md)
        self.stage4 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )
        # Final: 16 -> 3 channels, Sigmoid for [0, 1] output
        self.final = nn.Sequential(
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, STROKE_DIM) stroke params in [0, 1]
        Returns:
            (batch, 3, IMG_SIZE, IMG_SIZE) stroke image in [0, 1]
        """
        h = self.fc(x).view(-1, 128, 2, 2)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        return self.final(h)
```

**Critical note — stage 4 scale factor:** Use `scale_factor=4` in `stage4`, not `2`. Resolution path is 2→4→8→16→64. Using `scale_factor=2` throughout yields 32×32 output. Shape tests catch this immediately. (RESEARCH.md Pitfall 1, verified in-session.)

**No-BN constraint** (D-11): The class must have zero `nn.BatchNorm1d` / `nn.BatchNorm2d` instances. The test `test_neural_renderer_no_batchnorm` asserts this directly.

---

### `pretrain_renderer.py` (training script, batch + file-I/O)

**Analog:** `renderer.py` (import style, `@torch.no_grad()` usage) and `palette.py` (config import pattern, module-level constants).

**Imports pattern** (derived from `renderer.py` lines 1–5 and `palette.py` lines 1–2):
```python
# Source: CONTEXT.md D-04/D-05/D-06/D-07/D-12 + in-session verification (2026-06-09)
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # non-interactive — must appear before any other matplotlib import
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from config import IMG_SIZE, STROKE_DIM
from renderer import draw
from models.renderer import NeuralRenderer
```

Import order: stdlib → torch → third-party (matplotlib, numpy, tqdm) → project (config, renderer, models). `matplotlib.use('Agg')` must be called before `import matplotlib.pyplot`. (RESEARCH.md Pattern 5, verified in-session.)

**Module-level constants pattern** (`palette.py` lines 8–21 style — module-level tensors/constants defined before functions):
```python
TOTAL_PAIRS = 1_000_000
BATCH_SIZE = 1024
EXTREME_FRAC = 0.2
N_STEPS = TOTAL_PAIRS // BATCH_SIZE   # 976
VAL_EVERY = 50
VAL_N = 1000
```

**`@torch.no_grad()` guard pattern** (`renderer.py` line 10): All calls to the hard rasterizer `draw()` already carry `@torch.no_grad()`. In `pretrain_renderer.py`, wrap validation inference with `with torch.no_grad():` blocks — do NOT apply `@torch.no_grad()` to R's forward during training (gradients must flow for `loss.backward()`).

**Core training loop pattern** (RESEARCH.md Code Examples — Training Loop Skeleton):
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pre-generate val set once
val_params, val_targets = make_batch(VAL_N)
val_params = val_params.to(device)
val_targets = val_targets.to(device)

R = NeuralRenderer().to(device)
optimizer = torch.optim.Adam(R.parameters(), lr=1e-3)
# NOTE: no verbose=True — removed in PyTorch 2.x (RESEARCH.md Pitfall 2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

pbar = trange(N_STEPS, desc='Pretraining R')
for step in pbar:
    params, targets = make_batch(BATCH_SIZE)
    params = params.to(device)
    targets = targets.to(device)       # generate on CPU, move to GPU (RESEARCH.md Pitfall 3)

    preds = R(params)
    loss = nn.functional.mse_loss(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % VAL_EVERY == 0:
        with torch.no_grad():
            val_mse = nn.functional.mse_loss(R(val_params), val_targets).item()
        scheduler.step(val_mse)
        pbar.set_postfix(train=f'{loss.item():.5f}', val=f'{val_mse:.5f}',
                         lr=f"{optimizer.param_groups[0]['lr']:.2e}")
```

**Checkpoint save pattern** (RESEARCH.md Code Examples — Safe Checkpoint Save/Load):
```python
torch.save(R.state_dict(), 'renderer.pkl')
# NOT torch.save(R, ...) — full-module save is fragile across refactors
```

**Freeze-load pattern** (RESEARCH.md Pattern 3):
```python
def load_frozen_renderer(path: str, device: torch.device) -> NeuralRenderer:
    """
    Load pre-trained renderer R and freeze it.
    Both .eval() and requires_grad_(False) are required:
    - .eval() disables dropout/BN behavior at inference time
    - requires_grad_(False) prevents accidental gradient flow into R during RL training
    """
    R = NeuralRenderer()
    R.load_state_dict(torch.load(path, weights_only=True))   # weights_only=True: security (RESEARCH.md Security)
    R.eval()
    for p in R.parameters():
        p.requires_grad_(False)
    return R.to(device)
```

**Freeze-assertion pattern** (RESEARCH.md Pattern 4 — embed at bottom of `pretrain_renderer.py`):
```python
def param_norm(model: torch.nn.Module) -> float:
    return sum(p.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5

R_frozen = load_frozen_renderer('renderer.pkl', device=torch.device('cpu'))
checkpoint_norm = param_norm(R_frozen)
_ = R_frozen(torch.rand(1, STROKE_DIM))
assert abs(param_norm(R_frozen) - checkpoint_norm) < 1e-6, \
    f"R parameters changed after freeze: expected {checkpoint_norm:.8f}, got {param_norm(R_frozen):.8f}"
print(f"Freeze verified: param norm = {checkpoint_norm:.8f} (unchanged)")
```

**Visual gate pattern** (RESEARCH.md Pattern 5):
```python
VISUAL_TEST_CASES = [
    ('Thin H',      torch.tensor([0.5,  0.5,  0.3,  0.04, 0.0,  1.0, 0.0, 0.0])),
    ('Thin W',      torch.tensor([0.5,  0.5,  0.04, 0.3,  0.0,  0.0, 1.0, 0.0])),
    ('Tilted',      torch.tensor([0.5,  0.5,  0.3,  0.15, 0.45, 0.0, 0.0, 1.0])),
    ('Edge TL',     torch.tensor([0.05, 0.05, 0.2,  0.1,  0.0,  1.0, 0.5, 0.0])),
    ('Edge BR',     torch.tensor([0.95, 0.95, 0.2,  0.1,  0.0,  0.0, 1.0, 0.5])),
    ('Full canvas', torch.tensor([0.5,  0.5,  0.85, 0.85, 0.0,  1.0, 0.5, 0.0])),
    ('Full+tilted', torch.tensor([0.5,  0.5,  0.85, 0.85, 0.3,  0.5, 0.0, 1.0])),
    ('Extreme theta', torch.tensor([0.5, 0.5,  0.3,  0.1,  0.95, 1.0, 0.0, 0.0])),
]
# Thin-stroke h values are 0.04, above the subpixel boundary ~0.032 — intentional (RESEARCH.md Pitfall 4)

def save_visual_gate(R, path: str = 'visual_gate.png'):
    zeros = torch.zeros(3, IMG_SIZE, IMG_SIZE)
    n = len(VISUAL_TEST_CASES)
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i, (name, params) in enumerate(VISUAL_TEST_CASES):
        gt = draw(zeros, params).permute(1, 2, 0).numpy()
        axes[0][i].imshow(gt); axes[0][i].set_title(f'GT: {name}', fontsize=7); axes[0][i].axis('off')
        with torch.no_grad():
            pred = R(params.unsqueeze(0))[0].permute(1, 2, 0).cpu().numpy()
        axes[1][i].imshow(pred); axes[1][i].set_title(f'R: {name}', fontsize=7); axes[1][i].axis('off')
    axes[0][0].set_ylabel('Hard rasterizer')
    axes[1][0].set_ylabel('Neural R')
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f'Visual gate saved: {path}')
```

**Extreme-biasing data generator pattern** (RESEARCH.md Pattern 2):
```python
def sample_extreme_batch(n: int) -> torch.Tensor:
    params = torch.rand(n, STROKE_DIM)
    q = n // 4; r = n - 3 * q; i = 0
    params[i:i+q, 3] = torch.rand(q) * 0.05   # thin h
    i += q
    params[i:i+q, 2] = torch.rand(q) * 0.05   # thin w
    i += q
    params[i:i+q, 4] = 0.4 + torch.rand(q) * 0.6  # tilted
    i += q
    params[i:, 2] = 0.8 + torch.rand(r) * 0.2    # full-canvas w
    params[i:, 3] = 0.8 + torch.rand(r) * 0.2    # full-canvas h
    return params

def generate_targets(params: torch.Tensor) -> torch.Tensor:
    # Generate on CPU — loop-based rasterizer is faster on CPU (0.36s) than GPU (0.80s) for BS=1024
    # (RESEARCH.md Pitfall 3: verified benchmark 2026-06-09)
    zeros_canvas = torch.zeros(3, IMG_SIZE, IMG_SIZE)
    return torch.stack([draw(zeros_canvas, params[i]) for i in range(len(params))])
```

**Script entry point pattern** (project convention — standalone CLI script):
```python
if __name__ == '__main__':
    main()
```

---

### `tests/test_neural_renderer.py` (test, unit)

**Analog:** `tests/test_renderer.py` (exact match — same role, same assertion style)

**Imports pattern** (`tests/test_renderer.py` lines 1–3):
```python
import torch
import pytest
from models.renderer import NeuralRenderer
```
Copy structure directly. Replace `from renderer import draw` with `from models.renderer import NeuralRenderer`. Add `from config import IMG_SIZE, STROKE_DIM` for magic-number-free assertions.

**Shape assertion pattern** (`tests/test_renderer.py` lines 6–11):
```python
def test_neural_renderer_output_shape():
    R = NeuralRenderer()
    x = torch.rand(4, 8)
    out = R(x)
    assert out.shape == (4, 3, 64, 64), f"Expected (4,3,64,64), got {out.shape}"
```

**Range assertion pattern** (`tests/test_renderer.py` lines 42–46):
```python
def test_neural_renderer_output_range():
    R = NeuralRenderer()
    out = R(torch.rand(4, 8))
    assert out.min() >= 0.0 and out.max() <= 1.0
```

**No-autograd / requires_grad pattern** (`tests/test_renderer.py` lines 13–18):
```python
def test_draw_no_autograd():
    result = draw(canvas, params)
    assert not result.requires_grad
```
Equivalent for neural renderer (frozen inference):
```python
def test_neural_renderer_single_sample():
    R = NeuralRenderer()
    R.eval()
    for p in R.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        out = R(torch.rand(1, 8))
    assert out.shape == (1, 3, 64, 64)
    assert not out.requires_grad
```

**No-BatchNorm assertion pattern** (unique to neural renderer — no analog in Phase 1, copy from RESEARCH.md):
```python
def test_neural_renderer_no_batchnorm():
    R = NeuralRenderer()
    bn_layers = [m for m in R.modules()
                 if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
    assert len(bn_layers) == 0, f"Found BatchNorm layers: {bn_layers}"
```

**GPU skip pattern** (`tests/test_renderer.py` lines 59–66):
```python
def test_neural_renderer_gpu():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    device = torch.device("cuda")
    R = NeuralRenderer().to(device)
    out = R(torch.rand(1, 8, device=device))
    assert out.device.type == "cuda"
    assert out.shape == (1, 3, 64, 64)
```

**Freeze assertion test** (RESEARCH.md Pattern 4 — embed as pytest test, covers REND-03):
```python
def test_freeze_assertion():
    """Param norm must not change after frozen forward pass (REND-03)."""
    R = NeuralRenderer()
    # Simulate load_frozen_renderer without file I/O
    R.eval()
    for p in R.parameters():
        p.requires_grad_(False)

    def param_norm(m):
        return sum(p.data.norm(2).item() ** 2 for p in m.parameters()) ** 0.5

    norm_before = param_norm(R)
    with torch.no_grad():
        _ = R(torch.rand(4, 8))
    assert abs(param_norm(R) - norm_before) < 1e-6
```

---

## Shared Patterns

### Config import convention
**Source:** `renderer.py` line 5, `palette.py` line 3
**Apply to:** `models/renderer.py`, `pretrain_renderer.py`, `tests/test_neural_renderer.py`
```python
from config import IMG_SIZE, STROKE_DIM
```
Never use magic numbers (`64`, `8`) in any new file. Always import from `config`.

### `@torch.no_grad()` / `with torch.no_grad():` discipline
**Source:** `renderer.py` line 10 (`@torch.no_grad()` decorator on `draw`)
**Apply to:** `pretrain_renderer.py` (val loop), `tests/test_neural_renderer.py` (frozen inference tests)
- The hard rasterizer `draw()` already carries `@torch.no_grad()` — callers do not need to wrap it.
- R's forward during training must NOT be wrapped — gradients must flow.
- R's forward during validation and freeze-assertion tests must use `with torch.no_grad():`.

### Source-citation comment header
**Source:** `renderer.py` line 1
**Apply to:** All new files
```python
# Source: CONTEXT.md D-XX + in-session verification (2026-06-09)
```
Place on line 1 of each new file.

### Import of project root module from `models/` subdirectory
**Source:** Phase 1 D-07/D-08 conventions (`from config import ...`, `from renderer import draw`)
**Apply to:** `models/renderer.py`, `pretrain_renderer.py`
- Root module imports: `from renderer import draw`, `from config import IMG_SIZE`
- Package import: `from models.renderer import NeuralRenderer`
- Never add `models/` to `sys.path`. Always run from project root.

### `torch.load` security pattern
**Source:** RESEARCH.md Security Domain
**Apply to:** `pretrain_renderer.py` (freeze-verification step)
```python
torch.load(path, weights_only=True)
```
Always include `weights_only=True` when loading state dicts. Prevents arbitrary code execution via pickle.

### `ReduceLROnPlateau` — no `verbose=True`
**Source:** RESEARCH.md Pitfall 2 (verified in-session, PyTorch 2.10)
**Apply to:** `pretrain_renderer.py`
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
# Log LR manually if needed:
# print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
```
`verbose=True` raises `TypeError` in PyTorch 2.x. Do not use.

---

## No Analog Found

All three files have analogs. No gaps.

| File | Role | Data Flow | Analog Quality |
|------|------|-----------|----------------|
| `models/renderer.py` | model | transform | `renderer.py` — role-match (same rendering domain; different implementation paradigm: hard vs. neural) |
| `pretrain_renderer.py` | training script | batch + file-I/O | No standalone training script exists in Phase 1; patterns assembled from `renderer.py` (import style, `@no_grad` usage) + RESEARCH.md verified code examples |
| `tests/test_neural_renderer.py` | test | unit | `tests/test_renderer.py` — exact match |

---

## Metadata

**Analog search scope:** Project root (`renderer.py`, `config.py`, `palette.py`), `tests/` directory
**Files read:** `renderer.py`, `config.py`, `palette.py`, `tests/test_renderer.py`, `tests/test_palette.py`
**RESEARCH.md patterns consumed:** Pattern 1 (architecture), Pattern 2 (data generation), Pattern 3 (freeze-load), Pattern 4 (norm assertion), Pattern 5 (visual gate), all 6 Pitfalls
**Pattern extraction date:** 2026-06-09
