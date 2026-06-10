import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from models.actor import CoordConv


class TReLU(nn.Module):
    """
    Translated ReLU activation from Xiang & Li (2017).

    forward(x) = F.relu(x - alpha) + alpha
    where alpha is a scalar nn.Parameter initialized to 0.

    One learnable threshold per TReLU instance (scalar, NOT per-channel).
    Non-inplace implementation — inplace on a Parameter-derived tensor corrupts autograd.
    """

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Translated ReLU: relu(x - alpha) + alpha."""
        return F.relu(x - self.alpha) + self.alpha


class BasicBlockWN(nn.Module):
    """
    ResNet18 BasicBlock with WeightNorm + TReLU instead of BatchNorm + ReLU.

    Used in the critic backbone per paper §3.4 — BN is replaced by WN+TReLU
    to avoid BN's dependency on batch statistics (which hurts critic learning).

    WeightNorm applied via torch.nn.utils.parametrizations.weight_norm (modern
    deepcopy-safe API, NOT the deprecated torch.nn.utils.weight_norm).
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample=None):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv2d(in_channels, out_channels, 3,
                                           stride=stride, padding=1, bias=False))
        self.relu1 = TReLU()
        self.conv2 = weight_norm(nn.Conv2d(out_channels, out_channels, 3,
                                           stride=1, padding=1, bias=False))
        self.relu2 = TReLU()
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu2(out + identity)


class Critic(nn.Module):
    """
    DDPG model-based value network: ResNet18-like CNN with CoordConv stem
    and WeightNorm+TReLU backbone (no BatchNorm).

    Input:  (B, 7, 64, 64) — rendered next-state s_{t+1}
            = (canvas 3ch, target 3ch, step_normalized 1ch)
            NOTE: This is NOT a concatenated (state, action) vector.
            This is the model-based DDPG distinction (paper §3.3.1):
            the critic estimates V(s'), not Q(s, a).
    Output: (B, 1) scalar V(s'), UNBOUNDED — no sigmoid or tanh activation.

    Architecture:
      CoordConv(7->64, stride=2)  -> (B, 64, 32, 32)   [+TReLU after]
      Stage 1: 2x BasicBlockWN(64->64)              -> (B, 64,  32, 32)
      Stage 2: 2x BasicBlockWN(64->128,  stride=2)  -> (B, 128, 16, 16)
      Stage 3: 2x BasicBlockWN(128->256, stride=2)  -> (B, 256,  8,  8)
      Stage 4: 2x BasicBlockWN(256->512, stride=2)  -> (B, 512,  4,  4)
      AdaptiveAvgPool2d(1,1) -> flatten              -> (B, 512)
      Linear(512, 1)                                 -> (B, 1) unbounded

    WN+TReLU used instead of BatchNorm per paper §3.4 ("BN cannot speed up
    critic learning significantly"). The parametrizations.weight_norm API
    (not the deprecated torch.nn.utils.weight_norm) is mandatory for
    copy.deepcopy(critic) safety in ddpg/agent.py (Plan 03-04).

    CoordConv is imported from models/actor.py — not redefined here.
    """

    def __init__(self):
        super().__init__()
        self.coord_conv = CoordConv(7, 64, kernel_size=3, stride=2, padding=1)
        self.stem_relu = TReLU()
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)

    def _make_layer(self, in_ch: int, out_ch: int, blocks: int,
                    stride: int) -> nn.Sequential:
        """Build a critic residual stage with BasicBlockWN blocks.

        Downsample conv is also wrapped with weight_norm for consistency
        (per RESEARCH.md assumption A4).
        """
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                weight_norm(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)),
            )
        layers = [BasicBlockWN(in_ch, out_ch, stride=stride, downsample=downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlockWN(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 7, 64, 64) -> (B, 1) scalar V(s'), unbounded."""
        x = self.stem_relu(self.coord_conv(x))  # (B, 64, 32, 32)
        x = self.layer1(x)                       # (B, 64,  32, 32)
        x = self.layer2(x)                       # (B, 128, 16, 16)
        x = self.layer3(x)                       # (B, 256,  8,  8)
        x = self.layer4(x)                       # (B, 512,  4,  4)
        x = self.pool(x).flatten(1)              # (B, 512)
        return self.fc(x)                        # (B, 1) — no activation, unbounded
