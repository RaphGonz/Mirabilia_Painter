# Source: CONTEXT.md D-09/D-10/D-11 + in-session verification (2026-06-09)
# NeuralRenderer — differentiable CNN decoder. Input (batch, 8) → output (batch, 3, IMG_SIZE, IMG_SIZE).
# No BatchNorm (D-11). No Dropout. Frozen during RL training (see env.py load_frozen_renderer).
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
        self.fc = nn.Linear(STROKE_DIM, 1024)

        # Stage 1: 2x2 -> 4x4, 256 -> 64 channels
        self.stage1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 64, 3, padding=1),
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
        h = self.fc(x).view(-1, 256, 2, 2)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        return self.final(h)
