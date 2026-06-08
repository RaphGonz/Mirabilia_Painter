# Source: CONTEXT.md D-02..D-04 + in-session verification (2026-06-08)
# Hard rasterizer — opaque oriented rectangle via pure PyTorch tensor ops. No autograd. Pure tensor ops only.
import torch
import math
from config import IMG_SIZE

# MIN_VISIBLE_STROKE_WIDTH = 2.0 / (IMG_SIZE - 1)  # ~0.032 at 64x64 (comment only)


@torch.no_grad()
def draw(canvas: torch.Tensor, stroke_params: torch.Tensor) -> torch.Tensor:
    """
    Draw an opaque oriented rectangle onto canvas.

    Args:
        canvas: float32 tensor (3, H, W) in [0, 1]
        stroke_params: float32 tensor (8,) = (cx, cy, w, h, theta_01, r, g, b)
                       all values in [0, 1]; theta_01 maps to [0, pi]

    Returns:
        New canvas tensor (3, H, W), no autograd graph attached.
        Subpixel strokes (w or h < ~2/(H-1)) return the unmodified canvas — correct behavior.
    """
    cx, cy, w, h, theta_01, r, g, b = stroke_params
    theta = theta_01.item() * math.pi  # scalar via .item() — avoids TypeError from math.cos on tensor

    H_px, W_px = canvas.shape[-2], canvas.shape[-1]
    device = canvas.device

    # Pixel coordinate grid in [0, 1] x [0, 1]
    # linspace(0, 1, N): spacing = 1/(N-1), pixel centers at 0, 1/(N-1), ... 1
    ys = torch.linspace(0.0, 1.0, H_px, device=device)
    xs = torch.linspace(0.0, 1.0, W_px, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W) each — always pass indexing='ij'

    # Translate to rectangle-center-relative coordinates
    dx = grid_x - cx
    dy = grid_y - cy

    # Rotate pixel offsets into rectangle's local frame via 2x2 rotation matrix
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    local_x =  cos_t * dx + sin_t * dy   # along rectangle's length axis
    local_y = -sin_t * dx + cos_t * dy   # along rectangle's width axis

    # Boolean mask: pixel is inside rectangle (half-extent test)
    mask = (local_x.abs() <= w / 2.0) & (local_y.abs() <= h / 2.0)  # (H, W) bool
    mask = mask.unsqueeze(0)  # (1, H, W) for broadcast with (3, H, W) canvas

    # Paint: broadcast color over all pixels, select with mask
    color = torch.stack([r, g, b]).view(3, 1, 1).expand_as(canvas)
    return torch.where(mask, color, canvas)
