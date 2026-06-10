import argparse
import time
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import IMG_SIZE, STROKE_DIM, RENDERER_BETA
from renderer import draw
from models.renderer import SoftRasterizer

# ---------------------------------------------------------------------------
# Sampling helpers (retained for validation and Phase 3 env.py imports)
# ---------------------------------------------------------------------------

def sample_uniform_batch(n: int) -> torch.Tensor:
    return torch.rand(n, STROKE_DIM)


def sample_extreme_batch(n: int) -> torch.Tensor:
    """Sample from extreme regions: thin-h, thin-w, tilted, full-canvas."""
    params = torch.rand(n, STROKE_DIM)
    q = n // 4
    r = n - 3 * q
    i = 0
    params[i:i+q, 3] = torch.rand(q) * 0.05           # thin h
    i += q
    params[i:i+q, 2] = torch.rand(q) * 0.05           # thin w
    i += q
    params[i:i+q, 4] = 0.4 + torch.rand(q) * 0.6     # tilted
    i += q
    params[i:, 2] = 0.8 + torch.rand(r) * 0.2         # full-canvas
    params[i:, 3] = 0.8 + torch.rand(r) * 0.2
    return params


def generate_targets(params: torch.Tensor) -> torch.Tensor:
    """Hard rasterizer ground truth, CPU loop (faster than GPU for this task)."""
    zeros = torch.zeros(3, IMG_SIZE, IMG_SIZE)
    return torch.stack([draw(zeros, params[i]) for i in range(len(params))])


def make_batch(batch_size: int) -> tuple:
    """80% uniform + 20% extreme params with hard-rasterized targets."""
    n_extreme = int(batch_size * 0.35)
    n_uniform = batch_size - n_extreme
    params = torch.cat([sample_uniform_batch(n_uniform),
                        sample_extreme_batch(n_extreme)], dim=0)
    targets = generate_targets(params)
    return params, targets


# ---------------------------------------------------------------------------
# Freeze / load helpers (reused by Phase 3 env.py)
# ---------------------------------------------------------------------------

def load_frozen_renderer(path: str, device: torch.device) -> SoftRasterizer:
    """
    Load SoftRasterizer from checkpoint and freeze it.

    Both .eval() and requires_grad_(False) are kept for consistency with
    the Phase 3 interface, even though SoftRasterizer has no learnable params.
    weights_only=True prevents arbitrary code execution via pickle (T-02-PKL).
    """
    R = SoftRasterizer()
    R.load_state_dict(torch.load(path, weights_only=True))
    R.eval()
    for p in R.parameters():
        p.requires_grad_(False)
    return R.to(device)


def param_norm(model: nn.Module) -> float:
    """L2 norm across all learnable parameters (used for freeze verification)."""
    return sum(p.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5


# ---------------------------------------------------------------------------
# Visual gate
# ---------------------------------------------------------------------------

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


def save_visual_gate(R: SoftRasterizer, path: str = 'visual_gate.png') -> None:
    """2-row comparison figure: hard rasterizer (top) vs SoftRasterizer (bottom)."""
    zeros = torch.zeros(3, IMG_SIZE, IMG_SIZE)
    n = len(VISUAL_TEST_CASES)
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i, (name, params) in enumerate(VISUAL_TEST_CASES):
        gt = draw(zeros, params).permute(1, 2, 0).numpy()
        axes[0][i].imshow(gt)
        axes[0][i].set_title(f'GT: {name}', fontsize=7)
        axes[0][i].axis('off')

        with torch.no_grad():
            pred = R(params.unsqueeze(0))[0].permute(1, 2, 0).cpu().numpy()
        axes[1][i].imshow(pred)
        axes[1][i].set_title(f'R: {name}', fontsize=7)
        axes[1][i].axis('off')

    axes[0][0].set_ylabel('Hard rasterizer')
    axes[1][0].set_ylabel('SoftRasterizer')
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f'Visual gate saved: {path}')


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Validate SoftRasterizer against hard rasterizer, save renderer.pkl, run visual gate.

    No training required — SoftRasterizer is a closed-form analytical function.
    --beta controls edge softness in pixels (default 1.0, ~4px transition zone).

    Phase 4 env.py compositing formula (D-01):
        alpha = R.compute_alpha(params)              # (B, 1, H, W)
        stroke_rgb = R(params) / alpha.clamp(1e-6)  # unpremultiply
        new_canvas = alpha * stroke_rgb + (1 - alpha) * old_canvas
    """
    parser = argparse.ArgumentParser(description='Validate and save SoftRasterizer')
    parser.add_argument('--beta', type=float, default=RENDERER_BETA,
                        help=f'Edge softness in pixels (config default {RENDERER_BETA})')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  beta: {args.beta}')
    t0 = time.perf_counter()

    R = SoftRasterizer(beta=args.beta).to(device)

    # Validate against hard rasterizer on 2000 random pairs
    print('Validating soft rasterizer vs hard rasterizer (2000 pairs)...')
    val_params, val_targets = make_batch(2000)
    val_params  = val_params.to(device)
    val_targets = val_targets.to(device)
    fg_mask = val_targets > 0.01

    with torch.no_grad():
        val_preds = R(val_params)
        fg_mse  = (val_preds - val_targets).pow(2)[fg_mask].mean().item()
        all_mse = (val_preds - val_targets).pow(2).mean().item()

    elapsed = time.perf_counter() - t0
    print(f'Val fg-MSE  : {fg_mse:.6f}')
    print(f'Val full-MSE: {all_mse:.6f}')
    print(f'Elapsed     : {elapsed:.1f}s')

    if fg_mse < 0.05:
        print('Quality objective met: fg-MSE < 0.05')
    else:
        print(f'WARNING: fg-MSE {fg_mse:.4f} >= 0.05 — inspect visual_gate.png')

    # Save checkpoint (state_dict = registered buffers xx, yy + beta metadata)
    torch.save(R.state_dict(), 'renderer.pkl')
    print('Saved: renderer.pkl')

    # Load frozen and verify (T-02-NaN, T-02-FREEZE)
    cpu_device = torch.device('cpu')
    R_frozen = load_frozen_renderer('renderer.pkl', cpu_device)
    out = R_frozen(torch.rand(1, STROKE_DIM))
    assert out.isfinite().all(), 'FATAL: NaN/Inf in R output (T-02-NaN)'

    norm_before = param_norm(R_frozen)
    _ = R_frozen(torch.rand(1, STROKE_DIM))
    assert abs(param_norm(R_frozen) - norm_before) < 1e-6, 'FATAL: params changed after forward (T-02-FREEZE)'
    print(f'Freeze verified (no learnable params — param_norm = {norm_before:.6f})')

    # Visual gate
    save_visual_gate(R_frozen)


if __name__ == '__main__':
    main()
