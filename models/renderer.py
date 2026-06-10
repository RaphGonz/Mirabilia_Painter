import math
import torch
import torch.nn as nn
from config import IMG_SIZE, STROKE_DIM, RENDERER_BETA


class SoftRasterizer(nn.Module):
    """
    Differentiable soft rectangle rasterizer.

    Input:  (B, STROKE_DIM=8) stroke params in [0, 1] — (cx, cy, w, h, theta, r, g, b)
    Output: (B, 3, IMG_SIZE, IMG_SIZE) premultiplied (alpha * color) in [0, 1]

    Analytical formula — no training required.
    Uses sigmoid SDF approximation for differentiable edges:
        alpha(x,y) = sigmoid((w/2 - |dx'|) / beta) * sigmoid((h/2 - |dy'|) / beta)
    where (dx', dy') is (dx, dy) rotated by theta.

    beta controls edge softness in pixels:
        0.5 → ~2px transition (sharp)
        1.0 → ~4px transition (recommended)
        2.0 → ~9px transition (very soft)

    compute_alpha(params) exposes the alpha mask separately for env.py compositing.
    Frozen during RL training (eval + requires_grad=False in load_frozen_renderer).
    """

    def __init__(self, beta: float = RENDERER_BETA):
        super().__init__()
        self.beta = beta

        # Pixel coordinate grids — buffers move to device with .to(device)
        y = torch.arange(IMG_SIZE, dtype=torch.float32)
        x = torch.arange(IMG_SIZE, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('xx', xx.unsqueeze(0).contiguous())  # (1, H, W)
        self.register_buffer('yy', yy.unsqueeze(0).contiguous())  # (1, H, W)

    def _alpha(self, params: torch.Tensor) -> torch.Tensor:
        B = params.shape[0]
        cx    = params[:, 0] * IMG_SIZE
        cy    = params[:, 1] * IMG_SIZE
        w     = params[:, 2] * IMG_SIZE
        h     = params[:, 3] * IMG_SIZE
        theta = params[:, 4] * math.pi

        dx = self.xx - cx.view(B, 1, 1)
        dy = self.yy - cy.view(B, 1, 1)

        cos_a = torch.cos(theta).view(B, 1, 1)
        sin_a = torch.sin(theta).view(B, 1, 1)
        dx_r =  dx * cos_a + dy * sin_a
        dy_r = -dx * sin_a + dy * cos_a

        half_w = (w / 2).view(B, 1, 1)
        half_h = (h / 2).view(B, 1, 1)
        mask_x = torch.sigmoid((half_w - dx_r.abs()) / self.beta)
        mask_y = torch.sigmoid((half_h - dy_r.abs()) / self.beta)
        return mask_x * mask_y  # (B, H, W)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """(B, 8) → (B, 3, H, W) premultiplied stroke image."""
        alpha = self._alpha(params)  # (B, H, W)
        color = params[:, 5:8].view(params.shape[0], 3, 1, 1)
        return alpha.unsqueeze(1) * color  # (B, 3, H, W)

    def compute_alpha(self, params: torch.Tensor) -> torch.Tensor:
        """(B, 8) → (B, 1, H, W) alpha mask — for env.py compositing."""
        return self._alpha(params).unsqueeze(1)


# Backward-compat alias so Phase 3 and tests can import either name
NeuralRenderer = SoftRasterizer
