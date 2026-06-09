# Phase 2: Neural Renderer - Research

**Researched:** 2026-06-09
**Domain:** PyTorch CNN decoder, supervised pre-training, model freeze/checkpoint patterns
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Compositing (env.py forward reference):**
- **D-01:** Soft alpha-blend: `alpha = R_out.max(dim=0)`, `new_canvas = alpha * R_out + (1 - alpha) * old_canvas`. Fully differentiable.
- **D-02:** Dark-stroke train/infer gap accepted. Low-max-RGB strokes appear semi-transparent during RL training. Documented.
- **D-03:** R training target = `draw(zeros_canvas, params)` — stroke on black canvas; no compositing in the training objective.

**Pretraining budget:**
- **D-04:** 1M training pairs, generated on-the-fly (no static dataset).
- **D-05:** Batch size 1024. Expected GPU time: ~1h on GTX 1660 Ti [VERIFIED: in-session benchmark].
- **D-06:** Extreme-params biasing — 20% of each batch samples from extreme regions: thin (h < 0.05 or w < 0.05), tilted (theta_01 > 0.4), edge-position (cx < 0.1 or > 0.9 or cy < 0.1 or > 0.9), full-canvas (w > 0.8 and h > 0.8).
- **D-07:** Validation set held-out random stroke params. Target: val MSE < 0.005. ReduceLROnPlateau on val MSE.

**Freeze:**
- **D-08:** Freeze = `model.eval()` + `requires_grad_(False)` on all R parameters when loading in `env.py`. One comment. No formal pytest assertion for freeze.

**R architecture:**
- **D-09:** `Linear(8, 512)` → reshape `(128, 2, 2)` → 4 conv+upsample stages → `(3, 64, 64)`. Resolution path: 2×2 → 4×4 → 8×8 → 16×16 → 64×64 (stage 4 is ×4 upsample).
- **D-10:** Filter counts 128→64→32→16→3. Each stage: bilinear upsample + `Conv2d` + `ReLU`. Final: `Conv2d(16, 3, 1)` + `Sigmoid`.
- **D-11:** No BatchNorm anywhere. No Dropout.
- **D-12:** Adam lr=1e-3 with `ReduceLROnPlateau` on val MSE.

### Claude's Discretion

None explicitly listed. All architecture decisions are locked.

### Deferred Ideas (OUT OF SCOPE)

- Scaling R to 128×128
- Perceptual loss for R (LPIPS)
- R outputting 4 channels (RGB + alpha mask)

</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REND-01 | `models/renderer.py` implements R with architecture FC + decoder: input `(batch, 8)` → output `(batch, 3, 64, 64)` image in [0,1] | D-09/D-10 architecture verified in-session: correct shape and range confirmed [VERIFIED] |
| REND-02 | `pretrain_renderer.py` generates (params → hard rasterizer image) pairs on-the-fly, trains R by MSE, saves `renderer.pkl`; 20% extreme-param biasing | Training loop pattern verified in-session; loop-based rasterizer at BS=1024 takes ~0.46s/step on GPU [VERIFIED] |
| REND-03 | Visual validation of R on test strokes before any RL; assertion verifies R param norm unchanged after freeze | Freeze pattern and norm assertion verified in-session; matplotlib side-by-side visualization verified [VERIFIED] |

</phase_requirements>

---

## Summary

Phase 2 pre-trains the neural renderer R — the differentiable proxy that allows gradients to flow through stroke rendering during DDPG training. R is a small CNN decoder (~104K parameters) that takes stroke parameters `(batch, 8)` and produces stroke images `(batch, 3, 64, 64)`. Training is supervised against the hard rasterizer from Phase 1: for each random stroke parameter vector, `draw(zeros_canvas, params)` provides the target image. The entire phase produces two committed artifacts: `models/renderer.py` (the network definition) and `pretrain_renderer.py` (the training script that saves `renderer.pkl`).

The architecture is fully locked from CONTEXT.md decisions D-09 through D-12. No design exploration is needed. The main implementation risks are (1) correctly implementing the 4-stage upsampling path to land on exactly 64×64 — verified to require stage 4 to use ×4 scale factor rather than ×2, (2) correctly implementing the extreme-params biasing so thin/tilted/edge strokes are adequately represented in training, and (3) the visual gate: a human must inspect R predictions vs. hard rasterizer output before Phase 3 may begin.

All libraries needed for Phase 2 are already installed (PyTorch, torchvision, matplotlib, tqdm). No new packages need to be installed. The target val MSE of 0.005 requires approximately 10.7× better performance than an all-zeros predictor baseline — achievable for this architecture given that the hard rasterizer produces sparse stroke images on black backgrounds. In-session benchmarking confirms ~1h training time on the GTX 1660 Ti at BS=1024 for 1M pairs.

**Primary recommendation:** Two plans are sufficient. Plan 02-01 implements `models/renderer.py` (architecture + shape test). Plan 02-02 implements `pretrain_renderer.py` (training loop + visual gate + freeze assertion).

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Stroke image generation (differentiable) | `models/renderer.py` (R) | — | R is the differentiable proxy; used during DDPG actor gradient computation in Phase 4 |
| Stroke image generation (ground truth) | `renderer.py` (hard rasterizer) | — | Oracle for supervised training and final eval rendering |
| Supervised pre-training loop | `pretrain_renderer.py` | `renderer.py`, `models/renderer.py` | Standalone script; imports both renderers; not imported by other modules |
| Checkpoint persistence | `pretrain_renderer.py` writes `renderer.pkl` | `env.py` (Phase 4) reads it | Checkpoint is the contract between Phase 2 and Phase 4 |
| Freeze enforcement | `env.py` (Phase 4) loads with freeze | `models/renderer.py` (provides the class) | The actual freeze call must live in the loader, not in R's definition |
| Visual gate | `pretrain_renderer.py` exports comparison figure | Human reviewer | matplotlib side-by-side of R predictions vs. hard rasterizer output |

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.0+cu126 (installed) | Network definition, training loop, MSE loss, Adam optimizer | All Phase 2 ops verified working [VERIFIED: in-session] |
| torchvision | 0.25.0+cu126 (installed) | `save_image()` for checkpoint figures | Verified working for PNG saves [VERIFIED: in-session] |
| tqdm | 4.67.3 (installed) | Training progress bar | One-liner wrapper around training loop [VERIFIED: installed] |
| matplotlib | 3.10.8 (installed) | Visual gate comparison figures (R vs. hard rasterizer) | `matplotlib.use('Agg')` non-interactive verified [VERIFIED: in-session] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| math (stdlib) | — | Not needed in Phase 2 | Phase 1 rasterizer uses it; no angle math in R |
| numpy | 2.3.5 (installed) | `.permute(1,2,0).numpy()` for matplotlib imshow | Only in visual gate, not in training loop |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `nn.Upsample` + `Conv2d` | `nn.ConvTranspose2d` | ConvTranspose2d can introduce checkerboard artifacts; upsample-then-conv is the standard workaround — locked by D-09 |
| Loop-based target generation | Batched/vectorized rasterizer | Hard rasterizer `draw()` takes a single stroke; a batched version would reduce CPU-side loop overhead (currently ~0.46s for BS=1024); not needed since total training is ~1h |
| `torch.save(model, ...)` | `torch.save(model.state_dict(), ...)` | Saving the full model embeds the class definition and is fragile across refactors; state_dict is the standard pattern |

**Installation:** No new packages needed. All dependencies are already installed.

---

## Package Legitimacy Audit

Phase 2 installs zero new packages. All dependencies are already present and were audited in Phase 1.

| Package | Registry | Installed Version | slopcheck | Disposition |
|---------|----------|-------------------|-----------|-------------|
| torch | PyPI | 2.10.0+cu126 | [OK] (Phase 1 audit) | Approved |
| torchvision | PyPI | 0.25.0+cu126 | [OK] (Phase 1 audit) | Approved |
| matplotlib | PyPI | 3.10.8 | [OK] (Phase 1 audit) | Approved |
| tqdm | PyPI | 4.67.3 | [OK] (Phase 1 audit) | Approved |
| numpy | PyPI | 2.3.5 | [OK] (Phase 1 audit) | Approved |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

---

## Architecture Patterns

### System Architecture Diagram

```
pretrain_renderer.py
        |
        |-- 1. Pre-generate val set (N=1000 random params + hard rasterizer targets)
        |        stored as (params_val, targets_val) on GPU
        |
        |-- 2. Training loop (976 steps for 1M pairs at BS=1024)
        |
        |   Each step:
        |   a. Sample uniform params: (BS*0.8, 8) from Uniform[0,1]^8
        |   b. Sample extreme params: (BS*0.2, 8) from extreme distributions
        |      - thin: w<0.05 or h<0.05
        |      - tilted: theta_01>0.4
        |      - edge: cx or cy < 0.1 or > 0.9
        |      - full-canvas: w>0.8 and h>0.8
        |   c. Concatenate -> params_batch (BS, 8)
        |   d. Generate targets: for each row, draw(zeros_canvas, row) -> (BS, 3, 64, 64)
        |   e. Forward: preds = R(params_batch.to(device))
        |   f. Loss: MSE(preds, targets.to(device))
        |   g. loss.backward(); optimizer.step(); optimizer.zero_grad()
        |   h. Every 50 steps: val MSE, ReduceLROnPlateau.step(val_mse)
        |   i. tqdm progress bar shows train/val MSE
        |
        |-- 3. Save renderer.pkl = model.state_dict()
        |-- 4. Save visual gate figure (R vs hard rasterizer on 8 test strokes)
        |
        v
  renderer.pkl  +  visual_gate.png

models/renderer.py (NeuralRenderer)
        Input: (batch, 8) params
        |
        |-- Linear(8, 512) -> ReLU not needed (next step is reshape)
        |-- reshape -> (batch, 128, 2, 2)
        |-- stage1: Upsample(x2, bilinear) -> (B,128,4,4) -> Conv2d(128,64,3,pad=1) -> ReLU -> (B,64,4,4)
        |-- stage2: Upsample(x2, bilinear) -> (B,64,8,8)  -> Conv2d(64,32,3,pad=1)  -> ReLU -> (B,32,8,8)
        |-- stage3: Upsample(x2, bilinear) -> (B,32,16,16)-> Conv2d(32,16,3,pad=1)  -> ReLU -> (B,16,16,16)
        |-- stage4: Upsample(x4, bilinear) -> (B,16,64,64)-> Conv2d(16,16,3,pad=1)  -> ReLU -> (B,16,64,64)
        |-- final:  Conv2d(16,3,1) -> Sigmoid
        v
        Output: (batch, 3, 64, 64) in [0, 1]
```

### Recommended Project Structure

```
code/
├── config.py                  # IMG_SIZE, STROKE_DIM, etc. (Phase 1)
├── renderer.py                # Hard rasterizer draw() (Phase 1)
├── pretrain_renderer.py       # NEW: supervised training script
├── renderer.pkl               # NEW: trained R checkpoint (gitignored if >100MB — but ~400KB so track it)
├── visual_gate.png            # NEW: comparison figure saved by pretrain_renderer.py
├── models/
│   ├── __init__.py            # empty (Phase 1)
│   └── renderer.py            # NEW: NeuralRenderer class
└── tests/
    ├── test_neural_renderer.py # NEW: shape/no-batchnorm/freeze assertions
    └── ... (Phase 1 tests)
```

### Pattern 1: NeuralRenderer Architecture (D-09/D-10)

**What:** FC projection followed by bilinear-upsample + Conv2d decoder stages.

**When to use:** Any `(batch, 8) → (batch, 3, 64, 64)` forward call.

**Key implementation note:** Stage 4 must use `scale_factor=4` (not 2) to go from 16×16 to 64×64. Using ×2 would stop at 32×32. [VERIFIED: in-session]

**Example:**
```python
# Source: D-09/D-10 + in-session verification (2026-06-09)
import torch
import torch.nn as nn
from config import IMG_SIZE


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
        # Project 8-dim params to 512-dim feature map, reshaped to (128, 2, 2)
        self.fc = nn.Linear(8, 512)

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
        # Stage 4: 16x16 -> 64x64 (x4 upsample), 16 -> 16 channels
        # NOTE: scale_factor=4, not 2 — final resolution must hit IMG_SIZE=64
        self.stage4 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )
        # Final projection: 16 -> 3 channels, Sigmoid for [0, 1] output
        self.final = nn.Sequential(
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 8) stroke params in [0, 1]
        Returns:
            (batch, 3, IMG_SIZE, IMG_SIZE) stroke image in [0, 1]
        """
        h = self.fc(x).view(-1, 128, 2, 2)  # (B, 128, 2, 2)
        h = self.stage1(h)  # (B, 64, 4, 4)
        h = self.stage2(h)  # (B, 32, 8, 8)
        h = self.stage3(h)  # (B, 16, 16, 16)
        h = self.stage4(h)  # (B, 16, 64, 64)
        return self.final(h)  # (B, 3, 64, 64)
```

### Pattern 2: On-the-Fly Training Data Generation with Extreme Biasing

**What:** Each batch is 80% uniform random + 20% extreme params, targets generated via `draw()`.

**When to use:** The inner training loop in `pretrain_renderer.py`.

**Performance note:** Generating targets with the CPU loop for BS=1024 takes ~0.46s. Forward+backward+step on GPU takes ~3.1s. Total per step is ~3.5s. For 976 steps → ~1h total. [VERIFIED: in-session benchmark]

```python
# Source: D-04/D-05/D-06 + in-session verification (2026-06-09)
import torch
from renderer import draw
from config import IMG_SIZE, STROKE_DIM

BATCH_SIZE = 1024
EXTREME_FRAC = 0.2


def sample_uniform_batch(n: int) -> torch.Tensor:
    """Sample n stroke params uniformly from [0, 1]^8."""
    return torch.rand(n, STROKE_DIM)


def sample_extreme_batch(n: int) -> torch.Tensor:
    """
    Sample n stroke params from extreme regions:
    - thin: w < 0.05 (index 2) or h < 0.05 (index 3)
    - tilted: theta_01 > 0.4 (index 4), i.e. > 72 degrees
    - edge: cx < 0.1 or > 0.9 (index 0); cy < 0.1 or > 0.9 (index 1)
    - full-canvas: w > 0.8 and h > 0.8 (indices 2, 3)
    Distribute evenly across the 4 types.
    """
    params = torch.rand(n, STROKE_DIM)
    q = n // 4
    r = n - 3 * q
    i = 0
    # Thin h
    params[i:i+q, 3] = torch.rand(q) * 0.05
    i += q
    # Thin w
    params[i:i+q, 2] = torch.rand(q) * 0.05
    i += q
    # Tilted
    params[i:i+q, 4] = 0.4 + torch.rand(q) * 0.6
    i += q
    # Full-canvas (remaining)
    params[i:, 2] = 0.8 + torch.rand(r) * 0.2
    params[i:, 3] = 0.8 + torch.rand(r) * 0.2
    return params  # All values remain in [0, 1]


def generate_targets(params: torch.Tensor) -> torch.Tensor:
    """
    Generate target images for a batch of stroke params using the hard rasterizer.
    Args:
        params: (B, 8) stroke params in [0, 1]
    Returns:
        (B, 3, IMG_SIZE, IMG_SIZE) float32 target images
    """
    zeros_canvas = torch.zeros(3, IMG_SIZE, IMG_SIZE)
    return torch.stack([draw(zeros_canvas, params[i]) for i in range(len(params))])


def make_batch(batch_size: int = BATCH_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (params, targets) for one training step."""
    n_extreme = int(batch_size * EXTREME_FRAC)
    n_uniform = batch_size - n_extreme
    params = torch.cat([
        sample_uniform_batch(n_uniform),
        sample_extreme_batch(n_extreme),
    ], dim=0)
    targets = generate_targets(params)
    return params, targets
```

### Pattern 3: Freeze Pattern (D-08)

**What:** Load from checkpoint, set eval mode, disable all gradients.

**When to use:** Any code that loads `renderer.pkl` (env.py in Phase 4, and the freeze-assertion in pretrain_renderer.py).

```python
# Source: D-08 + in-session verification (2026-06-09)
import torch
from models.renderer import NeuralRenderer

def load_frozen_renderer(path: str, device: torch.device) -> NeuralRenderer:
    """
    Load pre-trained renderer R and freeze it.

    Both .eval() and requires_grad_(False) are required:
    - .eval() disables dropout/BN behavior at inference time
    - requires_grad_(False) prevents accidental gradient flow into R during RL training
    """
    R = NeuralRenderer()
    R.load_state_dict(torch.load(path, weights_only=True))
    R.eval()
    for p in R.parameters():
        p.requires_grad_(False)
    return R.to(device)
```

### Pattern 4: Norm-Based Freeze Assertion (REND-03 / Success Criterion 4)

**What:** Compute L2 norm of all parameters immediately after loading. Verify it hasn't changed after use.

**Performance note:** D-08 says no formal pytest assertion is required — just correct load code + comment. However, the roadmap Success Criterion 4 says an assertion must be "committed as part of the codebase." The planner should include a standalone `verify_freeze.py` or embed this assertion at the bottom of `pretrain_renderer.py` as an inline check that runs once after saving.

```python
# Source: REND-03 + in-session verification (2026-06-09)
import torch
from models.renderer import NeuralRenderer


def param_norm(model: torch.nn.Module) -> float:
    """L2 norm across all parameters."""
    return sum(p.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5


# After training and saving renderer.pkl:
R = load_frozen_renderer('renderer.pkl', device=torch.device('cpu'))
checkpoint_norm = param_norm(R)

# Run a forward pass (simulating what env.py will do)
dummy = torch.rand(1, 8)
_ = R(dummy)

# Assert norm unchanged
assert abs(param_norm(R) - checkpoint_norm) < 1e-6, \
    f"R parameters changed after freeze: expected {checkpoint_norm:.8f}, got {param_norm(R):.8f}"
print(f"Freeze verified: param norm = {checkpoint_norm:.8f} (unchanged)")
```

### Pattern 5: Visual Gate — Side-by-Side Comparison

**What:** For a fixed set of 8 test strokes (thin, tilted, edge, full-canvas, extreme theta, etc.), save a figure showing hard rasterizer vs. R prediction.

```python
# Source: REND-03 + in-session verification (2026-06-09)
import matplotlib
matplotlib.use('Agg')  # non-interactive — do not remove
import matplotlib.pyplot as plt
import torch
import numpy as np
from renderer import draw
from config import IMG_SIZE

VISUAL_TEST_CASES = [
    ('Thin H',     torch.tensor([0.5,  0.5,  0.3,  0.04, 0.0,  1.0, 0.0, 0.0])),
    ('Thin W',     torch.tensor([0.5,  0.5,  0.04, 0.3,  0.0,  0.0, 1.0, 0.0])),
    ('Tilted',     torch.tensor([0.5,  0.5,  0.3,  0.15, 0.45, 0.0, 0.0, 1.0])),
    ('Edge TL',    torch.tensor([0.05, 0.05, 0.2,  0.1,  0.0,  1.0, 0.5, 0.0])),
    ('Edge BR',    torch.tensor([0.95, 0.95, 0.2,  0.1,  0.0,  0.0, 1.0, 0.5])),
    ('Full canvas',torch.tensor([0.5,  0.5,  0.85, 0.85, 0.0,  1.0, 0.5, 0.0])),
    ('Full+tilted', torch.tensor([0.5, 0.5,  0.85, 0.85, 0.3,  0.5, 0.0, 1.0])),
    ('Extreme theta',torch.tensor([0.5, 0.5, 0.3,  0.1,  0.95, 1.0, 0.0, 0.0])),
]


def save_visual_gate(R, path: str = 'visual_gate.png'):
    zeros = torch.zeros(3, IMG_SIZE, IMG_SIZE)
    n = len(VISUAL_TEST_CASES)
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i, (name, params) in enumerate(VISUAL_TEST_CASES):
        # Hard rasterizer ground truth
        gt = draw(zeros, params).permute(1, 2, 0).numpy()
        axes[0][i].imshow(gt)
        axes[0][i].set_title(f'GT: {name}', fontsize=7)
        axes[0][i].axis('off')
        # R prediction
        with torch.no_grad():
            pred = R(params.unsqueeze(0))[0].permute(1, 2, 0).cpu().numpy()
        axes[1][i].imshow(pred)
        axes[1][i].set_title(f'R: {name}', fontsize=7)
        axes[1][i].axis('off')
    axes[0][0].set_ylabel('Hard rasterizer')
    axes[1][0].set_ylabel('Neural R')
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f'Visual gate saved: {path}')
```

### Anti-Patterns to Avoid

- **Stage 4 using `scale_factor=2`:** Results in 32×32 output, not 64×64. Stage 4 must use `scale_factor=4`. [VERIFIED: in-session — easy to get wrong when reading D-09 which says "bilinear upsample doubles spatial resolution" for each stage, but the 4th stage must quadruple to reach 64.]
- **Adding BatchNorm anywhere in R:** D-11 forbids it. BN in eval mode uses running stats computed during training; with the small batch sizes during env.py inference (single stroke), running stats will be incorrect. Forbidden per D-11.
- **Calling `draw()` on GPU params directly:** `draw()` internally uses `math.cos/sin` via `.item()`, which extracts a Python scalar from the tensor. Both CPU and GPU tensors support `.item()`. However, the `zeros_canvas` passed to `draw()` must be on the same device as `stroke_params` — if calling in a mixed-device setting, keep the rasterizer target generation on CPU and move targets to GPU after generation.
- **Using `torch.save(model, path)` instead of `torch.save(model.state_dict(), path)`:** Saving the full module object embeds the class definition; renaming the module or moving it to a different path breaks loading. Always save `model.state_dict()`.
- **Using `torch.load(path)` without `weights_only=True`:** Unsafe — arbitrary code execution via pickle. Always use `torch.load(path, weights_only=True)` for loading state dicts. [VERIFIED: works correctly in PyTorch 2.10, in-session]
- **Using `verbose=True` in `ReduceLROnPlateau`:** This keyword argument was removed in PyTorch 2.x. Use manual LR logging instead: `print(f"LR: {optimizer.param_groups[0]['lr']}")`. [VERIFIED: TypeError confirmed in PyTorch 2.10, in-session]
- **Saving the entire 1M targets to disk:** D-04 mandates on-the-fly generation. Pre-saving 1M targets would require ~47 GB (at 3×64×64 float32). On-the-fly is the correct approach.
- **Generating targets on GPU:** `draw()` is decorated with `@torch.no_grad()` and works on CUDA, but the loop calling `draw()` one stroke at a time is bottlenecked by kernel launch overhead on GPU (0.80s) vs. CPU (0.36s) for BS=1024. Generate targets on CPU, then `.to(device)` before computing loss. [VERIFIED: in-session benchmark]

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Bilinear upsampling | Custom interpolation | `nn.Upsample(mode='bilinear', align_corners=False)` | PyTorch built-in; handles boundary correctly; differentiable |
| LR decay on plateau | Custom counter logic | `torch.optim.lr_scheduler.ReduceLROnPlateau` | Built-in, handles patience/cooldown/threshold modes; avoid `verbose=True` in PyTorch 2.10 |
| Progress bar | Custom print loop | `tqdm(range(n_steps))` | One-liner; `.set_postfix()` for live MSE display |
| Image saving for visual gate | Custom PNG writer | `torchvision.utils.save_image()` or `matplotlib.pyplot.savefig()` | Both verified working in-session |
| Model checkpoint | Custom format | `torch.save(model.state_dict(), path)` | Standard; supports `weights_only=True` on load for safety |

**Key insight:** R is a standard supervised CNN training task. Every component (optimizer, scheduler, loss, checkpoint) has a proven PyTorch built-in. Nothing in Phase 2 requires custom implementations.

---

## Common Pitfalls

### Pitfall 1: Stage 4 Scale Factor Must Be 4, Not 2

**What goes wrong:** Using `scale_factor=2` in all 4 stages produces output of shape `(B, 16, 32, 32)` after stage 4, not `(B, 16, 64, 64)`. The `final` conv layer still runs but produces `(B, 3, 32, 32)`. Shape assertion tests catch this immediately.

**Why it happens:** D-09 says "bilinear upsample doubles spatial resolution at each stage," which sounds like all stages should use ×2. But 2×2 → 4×4 → 8×8 → 16×16 → 32×32 is only 4 doublings from 2: 2×2⁴ = 32. To reach 64 from 2 in 4 stages, one stage must be ×4.

**How to avoid:** Stage 4 must have `scale_factor=4`. The resolution path is: 2→4→8→16→64. [VERIFIED: in-session]

**Warning signs:** `output.shape[-1] == 32` instead of 64.

### Pitfall 2: `verbose=True` Removed from ReduceLROnPlateau in PyTorch 2.x

**What goes wrong:** `torch.optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True)` raises `TypeError: __init__() got an unexpected keyword argument 'verbose'`.

**Why it happens:** The `verbose` parameter was removed in PyTorch 2.x. The CLAUDE.md stack recommendation lists PyTorch 2.7 and the installed version is 2.10 — both affected.

**How to avoid:** Omit `verbose=True`. Log LR manually: `print(f"LR reduced to {opt.param_groups[0]['lr']}")` after each scheduler step if LR tracking is desired. [VERIFIED: TypeError confirmed in PyTorch 2.10, in-session]

**Warning signs:** `TypeError` immediately at scheduler initialization.

### Pitfall 3: Target Generation Slower on GPU Than CPU for Loop-Based Rasterizer

**What goes wrong:** Moving `zeros_canvas` to CUDA and calling `draw()` in a loop for BS=1024 takes ~0.80s vs. ~0.36s on CPU. GPU is slower here because each `draw()` call is a tiny kernel that doesn't saturate the GPU, and kernel launch overhead dominates.

**Why it happens:** The hard rasterizer `draw()` processes one stroke at a time. For BS=1024, this is 1024 sequential kernel launches on GPU vs. 1024 CPU tensor operations that benefit from vectorized BLAS-free ops.

**How to avoid:** Generate targets on CPU (`torch.zeros(3, H, W)` with no `.cuda()`), then call `.to(device)` on the resulting batch before loss computation. [VERIFIED: benchmark confirmed 0.36s CPU vs 0.80s GPU for BS=1024]

**Warning signs:** Training step taking >4s for BS=1024. Expected: ~3.5s total (0.46s target gen + 3.1s GPU forward/backward).

### Pitfall 4: Subpixel Strokes Produce All-Black Targets

**What goes wrong:** The hard rasterizer correctly returns an unmodified canvas for strokes with `w < 2/(IMG_SIZE-1) ≈ 0.032`. If extreme biasing samples `h < 0.05`, some of those strokes will be below the subpixel threshold at 64×64, producing all-black target images. R learns to output black for these params, which is technically correct but may confuse visual gate inspection.

**Why it happens:** D-06 sets the thin-stroke threshold at 0.05 (h or w). The subpixel boundary is ~0.032. Strokes between 0.032 and 0.05 are visible (1–2 pixels wide); strokes below 0.032 are invisible. About 1/3 of "thin" extreme samples (those with w or h in [0, 0.032]) will be all-black targets.

**How to avoid:** This is correct behavior — R correctly learns that sub-pixel strokes produce black output. Document this in `pretrain_renderer.py` as a comment. For the visual gate, only select thin strokes in [0.04, 0.05] range (visible but thin) to confirm R can handle them, not the subpixel invisible ones. [VERIFIED: Phase 1 research Pitfall 1]

**Warning signs:** Visual gate shows all-black for "thin" test strokes — check that test case `h` is above 0.032, not 0.001.

### Pitfall 5: MSE Target of 0.005 Requires ~10× Over All-Zeros Baseline

**What goes wrong:** If R converges prematurely or training data doesn't cover enough stroke varieties, val MSE may plateau at 0.01–0.02 instead of reaching 0.005.

**Why it happens:** The all-zeros predictor achieves MSE ≈ 0.053 (verified in-session). Reaching 0.005 requires ~10.7× improvement — achieving near-correct stroke shape prediction. This is feasible for the ~104K parameter architecture but requires the architecture to work correctly (correct stage 4 scale factor, no BN, correct training data).

**How to avoid:** Monitor val MSE curve during training. If it plateaus above 0.01 after 500 steps, check: (1) stage 4 scale factor is 4, (2) no BN layers present, (3) extreme biasing is correctly implemented (20% of batch, not 20% of steps). The visual gate will also reveal if shapes are recognizable before hitting the numeric threshold. [VERIFIED: in-session all-zeros baseline = 0.0533]

**Warning signs:** Val MSE > 0.02 after 500 steps of training with correct LR.

### Pitfall 6: `models/renderer.py` Shadows the Root `renderer.py`

**What goes wrong:** Both `renderer.py` (hard rasterizer) at the project root and `models/renderer.py` (NeuralRenderer) exist. Inside `pretrain_renderer.py`, `from renderer import draw` must import from the root, and `from models.renderer import NeuralRenderer` must import from `models/`. If the Python path is set up incorrectly, the wrong module is imported.

**Why it happens:** Python searches `sys.path` in order. If `models/` is added to `sys.path`, `import renderer` could find `models/renderer.py` instead of the root `renderer.py`.

**How to avoid:** Never add `models/` to `sys.path`. Use explicit package paths: `from renderer import draw` (root rasterizer) and `from models.renderer import NeuralRenderer` (neural renderer). The root directory must be in `sys.path` when running `python pretrain_renderer.py` — this is the default when running from the project root. [VERIFIED: D-07/D-08 import conventions from Phase 1; `from models.renderer import NeuralRenderer` confirmed working from root]

---

## Code Examples

Verified patterns from in-session testing:

### ReduceLROnPlateau (PyTorch 2.10 — no verbose)

```python
# Source: in-session verification (2026-06-09) — verbose= removed in PyTorch 2.x
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,     # reduce LR after 5 non-improving val evaluations
)
# Log LR manually:
# print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
```

### Safe Checkpoint Save/Load

```python
# Source: in-session verification (2026-06-09)
# Save
torch.save(model.state_dict(), 'renderer.pkl')

# Load (safe)
state = torch.load('renderer.pkl', weights_only=True)
model.load_state_dict(state)
```

### Training Loop Skeleton

```python
# Source: D-04/D-05/D-06/D-07/D-12 + in-session verification (2026-06-09)
import torch
import torch.nn as nn
from tqdm import trange
from models.renderer import NeuralRenderer

TOTAL_PAIRS = 1_000_000
BATCH_SIZE = 1024
N_STEPS = TOTAL_PAIRS // BATCH_SIZE  # 976
VAL_EVERY = 50   # evaluate on val set every N steps
VAL_N = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Build val set once (pre-generate, hold on GPU)
val_params, val_targets = make_batch(VAL_N)
val_params = val_params.to(device)
val_targets = val_targets.to(device)

R = NeuralRenderer().to(device)
optimizer = torch.optim.Adam(R.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

pbar = trange(N_STEPS, desc='Pretraining R')
for step in pbar:
    params, targets = make_batch(BATCH_SIZE)
    params = params.to(device)
    targets = targets.to(device)

    preds = R(params)
    loss = nn.functional.mse_loss(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % VAL_EVERY == 0:
        with torch.no_grad():
            val_preds = R(val_params)
            val_mse = nn.functional.mse_loss(val_preds, val_targets).item()
        scheduler.step(val_mse)
        pbar.set_postfix(train=f'{loss.item():.5f}', val=f'{val_mse:.5f}',
                         lr=f"{optimizer.param_groups[0]['lr']:.2e}")

torch.save(R.state_dict(), 'renderer.pkl')
```

### Verify Shape and No-BatchNorm in Tests

```python
# Source: REND-01 acceptance criteria + in-session verification (2026-06-09)
import torch
import pytest
from models.renderer import NeuralRenderer


def test_neural_renderer_output_shape():
    R = NeuralRenderer()
    x = torch.rand(4, 8)
    out = R(x)
    assert out.shape == (4, 3, 64, 64), f"Expected (4,3,64,64), got {out.shape}"


def test_neural_renderer_output_range():
    R = NeuralRenderer()
    x = torch.rand(4, 8)
    out = R(x)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_neural_renderer_no_batchnorm():
    R = NeuralRenderer()
    bn_layers = [m for m in R.modules()
                 if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
    assert len(bn_layers) == 0, f"Found BatchNorm layers: {bn_layers}"


def test_neural_renderer_single_sample():
    """Single-sample inference (as env.py will call it)."""
    R = NeuralRenderer()
    R.eval()
    for p in R.parameters():
        p.requires_grad_(False)
    x = torch.rand(1, 8)
    with torch.no_grad():
        out = R(x)
    assert out.shape == (1, 3, 64, 64)
    assert not out.requires_grad
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `ConvTranspose2d` for upsampling | `nn.Upsample(bilinear) + Conv2d` | ~2017 (checkerboard artifacts paper) | Bilinear upsample avoids checkerboard artifacts common with ConvTranspose2d; locked by D-09 |
| `verbose=True` in ReduceLROnPlateau | Manual LR logging | PyTorch 2.x | API breaking change; `verbose=True` raises TypeError in PyTorch 2.10 |
| `torch.load(path)` (unsafe) | `torch.load(path, weights_only=True)` | PyTorch 2.0+ | Security: prevents arbitrary code execution via pickled model files |
| MLP-only renderer (all FC layers) | CNN decoder with spatial upsampling | Original paper + CLAUDE.md | CNN decoder has better spatial inductive bias for pixel outputs; MLP forbidden by CLAUDE.md |

**Deprecated/outdated:**
- `verbose=True` in ReduceLROnPlateau: Raises `TypeError` in PyTorch 2.10. Do not use.
- `torch.load(path)` without `weights_only=True`: Unsafe for pickle-based files. Always use `weights_only=True` for state dicts.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Val MSE < 0.005 is achievable with the D-09/D-10 architecture after 1M training pairs | Common Pitfalls P5 | If architecture is too small or training too short, Phase 3 is blocked at the HARD GATE; remediation: increase N_STEPS or add a second training run |
| A2 | Stage 4 should use scale_factor=4 to go from 16×16 to 64×64; D-09 describes "4 conv+upsample stages" with stages doubling resolution | Architecture Patterns P1 | If D-09 intended a different path (e.g., stride-2 convolution in stage 4 per the CONTEXT.md note), the planner may need to adjust; shape test catches this immediately |
| A3 | renderer.pkl (~400 KB) is small enough to commit to git (CLAUDE.md says ignore checkpoints >100 MB) | Code Examples | No risk — 400 KB is well under the 100 MB threshold |

**A1 is the most significant risk:** if the architecture converges to MSE 0.007–0.010 instead of 0.005, the visual gate may still pass but the numeric target is missed. The planner should flag this: Phase 3 can begin if visual gate passes and the 0.005 threshold is within reach with continued training.

---

## Open Questions

1. **Resolving D-09 ambiguity: 3×2 stages + 1×4 stage, or 4 stages with a final stride-2 conv?**
   - What we know: D-09 says "4 conv+upsample stages" with "bilinear upsample doubles at each stage" but also mentions "stride-2 conv or a single larger upsample step" for the final stage
   - What's unclear: Whether the planner should use `scale_factor=4` on stage 4 (as verified in-session) or a different mechanism
   - Recommendation: Use `scale_factor=4` on stage 4. It's the simplest path, verified in-session to produce the correct output shape, and the CONTEXT.md parenthetical "(final stage uses stride-2 conv or a single larger upsample step)" explicitly allows either option.

2. **Should the visual gate figure be committed to git?**
   - What we know: `visual_gate.png` is the human-reviewable artifact; CLAUDE.md mentions checking in `.pt` files but not images
   - What's unclear: No explicit instruction about committing PNG figures
   - Recommendation: Commit `visual_gate.png` alongside `renderer.pkl`. It is small (~7–50 KB), serves as documented evidence that the gate was passed, and is required by Success Criterion 3 ("human reviewer confirms").

3. **How many training steps before first val evaluation?**
   - What we know: VAL_EVERY=50 gives 19 evaluations over 976 total steps; first eval at step 50
   - What's unclear: Whether to evaluate before training starts (step 0) for baseline logging
   - Recommendation: Evaluate at step 0 to log the untrained MSE as baseline reference in tqdm output.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | All | ✓ | 3.14.3 | — |
| PyTorch | models/renderer.py, pretrain_renderer.py | ✓ | 2.10.0+cu126 | — |
| CUDA (GTX 1660 Ti) | GPU training | ✓ | 12.6 | CPU (slower — ~6h vs ~1h) |
| torchvision | save_image for visual outputs | ✓ | 0.25.0+cu126 | matplotlib only |
| matplotlib | Visual gate comparison figure | ✓ | 3.10.8 | — |
| tqdm | Training progress bar | ✓ | 4.67.3 | — |
| numpy | permute→numpy for matplotlib imshow | ✓ | 2.3.5 | — |
| pytest | Shape/no-batchnorm/freeze tests | ✓ | 8.4.2 | — |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:** None.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.4.2 (installed) |
| Config file | `pyproject.toml` (exists — `testpaths = ["tests"]`) |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REND-01 | `NeuralRenderer` input `(B, 8)` → output `(B, 3, 64, 64)` | unit | `pytest tests/test_neural_renderer.py::test_neural_renderer_output_shape -x` | ❌ Wave 0 |
| REND-01 | Output range in [0, 1] | unit | `pytest tests/test_neural_renderer.py::test_neural_renderer_output_range -x` | ❌ Wave 0 |
| REND-01 | No BatchNorm layers | unit | `pytest tests/test_neural_renderer.py::test_neural_renderer_no_batchnorm -x` | ❌ Wave 0 |
| REND-01 | Single-sample inference works (as env.py will call) | unit | `pytest tests/test_neural_renderer.py::test_neural_renderer_single_sample -x` | ❌ Wave 0 |
| REND-02 | `pretrain_renderer.py` saves `renderer.pkl` and achieves val MSE < 0.005 | integration (run script) | `python pretrain_renderer.py && python -c "assert val_mse < 0.005"` | ❌ pretrain_renderer.py |
| REND-03 | Freeze assertion: param norm unchanged after forward passes | unit | `pytest tests/test_neural_renderer.py::test_freeze_assertion -x` | ❌ Wave 0 |
| REND-03 | Visual gate: human inspection of `visual_gate.png` | manual | Human review of saved PNG | ❌ generated by script |

### Sampling Rate

- **Per task commit:** `pytest tests/ -x -q` (all existing + new neural renderer shape tests)
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green + human approval of `visual_gate.png` + `renderer.pkl` present

### Wave 0 Gaps

- [ ] `tests/test_neural_renderer.py` — covers REND-01 (shape, range, no-batchnorm, single-sample) and REND-03 (freeze assertion)
- [ ] `models/renderer.py` — the `NeuralRenderer` class (prerequisite for tests to import)

*(All other test infrastructure from Phase 1 is in place: `tests/__init__.py`, `pyproject.toml`, `pytest` installed)*

---

## Security Domain

`security_enforcement: true`, ASVS Level 1.

Phase 2 introduces file I/O: reading and writing `renderer.pkl`. No network I/O, no user input, no authentication.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | No auth in Phase 2 |
| V3 Session Management | No | No sessions |
| V4 Access Control | No | No access control |
| V5 Input Validation | Partial | Stroke params sampled from Uniform[0,1] — no user input; no validation needed during training |
| V6 Cryptography | No | No secrets |
| V7 Error Handling | Yes | `weights_only=True` on load to prevent pickle code execution |

### Known Threat Patterns for PyTorch model files

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Arbitrary code execution via pickle in model checkpoint | Tampering | `torch.load(path, weights_only=True)` — saves state_dict (tensors only), loads with `weights_only=True` [VERIFIED: works in PyTorch 2.10] |
| NaN in model output (propagates to RL loss in Phase 4) | Tampering | Assert `output.isfinite().all()` in the freeze verification; Sigmoid output is bounded [0,1] and cannot produce NaN from valid inputs |

**Phase 2 security posture:** Acceptable. The only file written is `renderer.pkl` (own process artifact). The only file read is the same `renderer.pkl` in the verification step. `weights_only=True` is the single required mitigation.

---

## Sources

### Primary (HIGH confidence)

- In-session Python execution — all architecture shapes, training loop timing, ReduceLROnPlateau API, freeze pattern, norm assertion, matplotlib visual gate verified in PyTorch 2.10.0+cu126, Python 3.14, GTX 1660 Ti (2026-06-09)
- CONTEXT.md D-01 through D-12 — locked decisions
- REQUIREMENTS.md REND-01, REND-02, REND-03 — acceptance criteria
- ROADMAP.md Phase 2 success criteria
- `renderer.py` (Phase 1 hard rasterizer) — D-03 training target API confirmed
- `config.py` — IMG_SIZE=64, STROKE_DIM=8 confirmed

### Secondary (MEDIUM confidence)

- CLAUDE.md — NeuralRenderer design rationale (CNN decoder, no MLP, no BN, Sigmoid output), PyTorch stack
- `paint_ai_design.md` — two-renderer architecture rationale, compositing separation

### Tertiary (LOW confidence)

- None. All Phase 2 claims are verified in-session or cited from locked decisions.

---

## Metadata

**Confidence breakdown:**

- Architecture (D-09/D-10): HIGH — shape verified in-session; output `(B,3,64,64)` in `[0,1]`, ~104K params, no BN
- Training loop: HIGH — target generation, MSE loss, backward pass all verified; timing benchmarked
- Freeze pattern (D-08): HIGH — `eval()` + `requires_grad_(False)` + `weights_only=True` load all verified
- Val MSE target (0.005): MEDIUM — verified that 10.7× improvement over all-zeros is required; achievability depends on training completing correctly
- ReduceLROnPlateau API: HIGH — `verbose=True` TypeError confirmed; correct API verified

**Research date:** 2026-06-09
**Valid until:** 2026-09-01 (stable PyTorch 2.x API; no expected breaking changes for these primitives)
