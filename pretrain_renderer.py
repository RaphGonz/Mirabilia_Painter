# Source: CONTEXT.md D-04/D-05/D-06/D-07/D-12 + in-session verification (2026-06-09)
import argparse
import datetime
import json
import time
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

# ---------------------------------------------------------------------------
# Module-level constants (D-04/D-05/D-06/D-07)
# ---------------------------------------------------------------------------
TOTAL_PAIRS = 1_000_000
BATCH_SIZE = 1024
EXTREME_FRAC = 0.2
N_STEPS = TOTAL_PAIRS // BATCH_SIZE   # 976
QUICK_STEPS = 200                      # autoresearch budget (~3-5 min on local GPU)
VAL_EVERY = 50
VAL_N = 1000


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

def sample_uniform_batch(n: int) -> torch.Tensor:
    """Sample n stroke params uniformly from [0, 1]^STROKE_DIM."""
    return torch.rand(n, STROKE_DIM)


def sample_extreme_batch(n: int) -> torch.Tensor:
    """
    Sample n stroke params from extreme regions (D-06):
    - thin h: index 3 in [0, 0.05)
    - thin w: index 2 in [0, 0.05)
    - tilted: index 4 (theta_01) in [0.4, 1.0)  — > 72 degrees
    - full-canvas: indices 2 and 3 in [0.8, 1.0)
    Each category receives ~n/4 samples. All values remain in [0, 1].
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
    # Full-canvas (remaining samples)
    params[i:, 2] = 0.8 + torch.rand(r) * 0.2
    params[i:, 3] = 0.8 + torch.rand(r) * 0.2
    return params


def generate_targets(params: torch.Tensor) -> torch.Tensor:
    """
    Generate target images for a batch of stroke params using the hard rasterizer.

    Args:
        params: (B, STROKE_DIM) stroke params in [0, 1], on CPU

    Returns:
        (B, 3, IMG_SIZE, IMG_SIZE) float32 target images, on CPU

    NOTE: Target generation runs on CPU — the loop-based rasterizer is faster on CPU (~0.36s)
    than on GPU (~0.80s) for BS=1024 due to kernel launch overhead (RESEARCH.md Pitfall 3).
    Move the resulting tensor to GPU after generation.

    NOTE: Sub-pixel strokes (w or h < ~0.032 at 64x64) yield all-black targets — this is
    correct behavior. R learns that sub-pixel params produce black output (RESEARCH.md Pitfall 4).
    """
    zeros_canvas = torch.zeros(3, IMG_SIZE, IMG_SIZE)
    return torch.stack([draw(zeros_canvas, params[i]) for i in range(len(params))])


def make_batch(batch_size: int = BATCH_SIZE) -> tuple:
    """
    Generate one training batch: 80% uniform + 20% extreme params, with targets.

    Returns:
        (params, targets): both CPU tensors
        - params: (batch_size, STROKE_DIM) in [0, 1]
        - targets: (batch_size, 3, IMG_SIZE, IMG_SIZE) in [0, 1]
    """
    n_extreme = int(batch_size * EXTREME_FRAC)
    n_uniform = batch_size - n_extreme
    params = torch.cat([
        sample_uniform_batch(n_uniform),
        sample_extreme_batch(n_extreme),
    ], dim=0)
    targets = generate_targets(params)
    return params, targets


# ---------------------------------------------------------------------------
# Freeze verification helpers
# ---------------------------------------------------------------------------

def load_frozen_renderer(path: str, device: torch.device) -> NeuralRenderer:
    """
    Load pre-trained renderer R from checkpoint and freeze it.

    Both .eval() and requires_grad_(False) are required (D-08):
    - .eval() disables dropout/BN behavior at inference time
    - requires_grad_(False) prevents accidental gradient flow into R during RL training

    Security: weights_only=True prevents arbitrary code execution via pickle (T-02-PKL, ASVS V7).
    """
    R = NeuralRenderer()
    R.load_state_dict(torch.load(path, weights_only=True))
    R.eval()
    for p in R.parameters():
        p.requires_grad_(False)
    return R.to(device)


def param_norm(model: torch.nn.Module) -> float:
    """L2 norm across all parameters (used for freeze verification, REND-03)."""
    return sum(p.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5


# ---------------------------------------------------------------------------
# Visual gate constants and function
# ---------------------------------------------------------------------------

# 8 named test cases for the visual gate (RESEARCH.md Pattern 5 / REND-03)
# Thin-stroke h/w values are 0.04 — above the ~0.032 sub-pixel boundary at 64x64 (Pitfall 4)
VISUAL_TEST_CASES = [
    ('Thin H',        torch.tensor([0.5,  0.5,  0.3,  0.04, 0.0,  1.0, 0.0, 0.0])),
    ('Thin W',        torch.tensor([0.5,  0.5,  0.04, 0.3,  0.0,  0.0, 1.0, 0.0])),
    ('Tilted',        torch.tensor([0.5,  0.5,  0.3,  0.15, 0.45, 0.0, 0.0, 1.0])),
    ('Edge TL',       torch.tensor([0.05, 0.05, 0.2,  0.1,  0.0,  1.0, 0.5, 0.0])),
    ('Edge BR',       torch.tensor([0.95, 0.95, 0.2,  0.1,  0.0,  0.0, 1.0, 0.5])),
    ('Full canvas',   torch.tensor([0.5,  0.5,  0.85, 0.85, 0.0,  1.0, 0.5, 0.0])),
    ('Full+tilted',   torch.tensor([0.5,  0.5,  0.85, 0.85, 0.3,  0.5, 0.0, 1.0])),
    ('Extreme theta', torch.tensor([0.5,  0.5,  0.3,  0.1,  0.95, 1.0, 0.0, 0.0])),
]


def save_visual_gate(R: NeuralRenderer, path: str = 'visual_gate.png') -> None:
    """
    Save a 2-row comparison figure: hard rasterizer GT (top) vs Neural R (bottom).

    Args:
        R: frozen NeuralRenderer (should be in eval mode, requires_grad=False)
        path: output path for the figure
    """
    zeros = torch.zeros(3, IMG_SIZE, IMG_SIZE)
    n = len(VISUAL_TEST_CASES)
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i, (name, params) in enumerate(VISUAL_TEST_CASES):
        # Hard rasterizer ground truth
        gt = draw(zeros, params).permute(1, 2, 0).numpy()
        axes[0][i].imshow(gt)
        axes[0][i].set_title(f'GT: {name}', fontsize=7)
        axes[0][i].axis('off')
        # Neural R prediction
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


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Supervised pre-training of NeuralRenderer R.

    Modes (--mode):
      quick  QUICK_STEPS=200 steps, outputs latest_result.json, skips renderer.pkl + visual gate.
             Used by the autoresearch loop to compare candidate configs quickly (~3-5 min).
      full   N_STEPS=976 steps (1M pairs), saves renderer.pkl, runs freeze verification + visual gate.
             Use after autoresearch identifies the best config.

    Pipeline (full):
    1. Pre-generate validation set (VAL_N random pairs) on CPU, move to device
    2. Train R for N_STEPS steps — each step: 80% uniform + 20% extreme-param batch
    3. Save R.state_dict() to renderer.pkl
    4. Load renderer.pkl as frozen R (weights_only=True — T-02-PKL)
    5. Assert finite output (T-02-NaN)
    6. Assert param norm unchanged after frozen forward (T-02-FREEZE / REND-03)
    7. Save visual_gate.png for human inspection

    Phase 4 env.py compositing formula (D-01 — documented here for Phase 4 executor):
        alpha = R_out.max(dim=0).values       # per-pixel max across RGB channels
        new_canvas = alpha * R_out + (1 - alpha) * old_canvas
    Dark-stroke train/infer gap (D-02): strokes with low max-RGB appear semi-transparent
    during RL training (soft alpha blend), but are fully opaque at inference (hard rasterizer).
    This is accepted behavior and is documented in paint_ai_design.md.
    """
    parser = argparse.ArgumentParser(description='Pre-train NeuralRenderer R')
    parser.add_argument(
        '--mode', choices=['quick', 'full'], default='full',
        help='quick: 200 steps, writes latest_result.json; full: 976 steps, saves renderer.pkl',
    )
    args = parser.parse_args()
    n_steps = QUICK_STEPS if args.mode == 'quick' else N_STEPS

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  mode: {args.mode}  |  steps: {n_steps}')
    t0 = time.perf_counter()

    # Pre-generate validation set once (held-out, not used in training)
    print(f'Generating validation set ({VAL_N} pairs) ...')
    val_params, val_targets = make_batch(VAL_N)
    val_params = val_params.to(device)
    val_targets = val_targets.to(device)

    # Build model and optimizer
    R = NeuralRenderer().to(device)
    optimizer = torch.optim.Adam(R.parameters(), lr=1e-3)
    # NOTE: verbose keyword arg removed in PyTorch 2.x (RESEARCH.md Pitfall 2 — raises TypeError)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Compute baseline val MSE (untrained R) before training starts
    with torch.no_grad():
        baseline_val_mse = nn.functional.mse_loss(R(val_params), val_targets).item()
    print(f'Baseline val MSE (untrained R): {baseline_val_mse:.5f}')

    # Training loop
    pbar = trange(n_steps, desc=f'Pretraining R [{args.mode}]')
    for step in pbar:
        params, targets = make_batch(BATCH_SIZE)
        params = params.to(device)
        targets = targets.to(device)   # generate on CPU, move to GPU (RESEARCH.md Pitfall 3)

        # Forward + loss — do NOT wrap in torch.no_grad() here; gradients must flow
        preds = R(params)
        loss = nn.functional.mse_loss(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation and LR scheduling
        if step % VAL_EVERY == 0:
            with torch.no_grad():
                val_mse = nn.functional.mse_loss(R(val_params), val_targets).item()
            scheduler.step(val_mse)
            pbar.set_postfix(
                train=f'{loss.item():.5f}',
                val=f'{val_mse:.5f}',
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

    # Final validation MSE
    with torch.no_grad():
        final_val_mse = nn.functional.mse_loss(R(val_params), val_targets).item()
    elapsed = time.perf_counter() - t0
    print(f'\nTraining complete. Final val MSE: {final_val_mse:.6f}  ({elapsed:.1f}s)')

    # ---------------------------------------------------------------------------
    # Quick mode: write machine-readable result and exit (no renderer.pkl)
    # ---------------------------------------------------------------------------
    if args.mode == 'quick':
        result = {
            'mode': 'quick',
            'steps': n_steps,
            'final_val_mse': round(final_val_mse, 8),
            'baseline_val_mse': round(baseline_val_mse, 8),
            'elapsed_seconds': round(elapsed, 1),
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        }
        with open('latest_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        print(f'Result written: latest_result.json  (val MSE {final_val_mse:.6f})')
        return

    # ---------------------------------------------------------------------------
    # Full mode: checkpoint, verification, visual gate
    # ---------------------------------------------------------------------------
    if final_val_mse < 0.005:
        print('Target met: val MSE < 0.005 (REND-02 / ROADMAP Success Criterion 2)')
    else:
        print(
            f'WARNING: val MSE {final_val_mse:.6f} >= 0.005 target (RESEARCH.md A1). '
            'Visual gate may still pass if shapes are recognizable. Flag in SUMMARY.'
        )

    # Save checkpoint — state_dict only (never torch.save(R, ...) — fragile across refactors)
    torch.save(R.state_dict(), 'renderer.pkl')
    print('Saved: renderer.pkl')

    # Load frozen R and verify finite output (T-02-NaN)
    cpu_device = torch.device('cpu')
    R_frozen = load_frozen_renderer('renderer.pkl', cpu_device)
    out = R_frozen(torch.rand(1, STROKE_DIM))
    assert out.isfinite().all(), \
        'FATAL: R output contains NaN or Inf after load — checkpoint may be corrupted (T-02-NaN)'

    # Freeze verification: param norm must not change after a forward pass (T-02-FREEZE / REND-03)
    checkpoint_norm = param_norm(R_frozen)
    _ = R_frozen(torch.rand(1, STROKE_DIM))
    assert abs(param_norm(R_frozen) - checkpoint_norm) < 1e-6, (
        f'R parameters changed after freeze: expected {checkpoint_norm:.8f}, '
        f'got {param_norm(R_frozen):.8f}'
    )
    print(f'Freeze verified: param norm = {checkpoint_norm:.8f} (unchanged after forward pass)')

    # Visual gate: save GT vs R comparison for human review (REND-03 / Success Criterion 3)
    save_visual_gate(R_frozen)


if __name__ == '__main__':
    main()
