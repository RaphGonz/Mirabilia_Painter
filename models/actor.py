import torch
import torch.nn as nn
import torch.nn.functional as F
from config import IMG_SIZE, STROKE_DIM, STROKES_PER_STEP


class CoordConv(nn.Module):
    """
    Wraps a Conv2d to prepend normalized (x, y) coordinate channels.

    Input:  (B, in_channels, H, W)
    Output: (B, out_channels, H', W')  [H', W' depend on stride/padding]

    The inner Conv2d receives (in_channels + 2) input channels.
    Coordinate grids are registered as buffers so they follow .to(device).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 bias: bool = False):
        super().__init__()
        H = W = IMG_SIZE
        # xx: x-coord normalized to [-1, 1], shape (1, 1, H, W)
        xx = torch.linspace(-1, 1, W).view(1, 1, 1, W).expand(1, 1, H, W)
        # yy: y-coord normalized to [-1, 1], shape (1, 1, H, W)
        yy = torch.linspace(-1, 1, H).view(1, 1, H, 1).expand(1, 1, H, W)
        self.register_buffer('xx', xx.contiguous())
        self.register_buffer('yy', yy.contiguous())
        # Inner Conv2d receives in_channels + 2 input channels
        self.conv = nn.Conv2d(in_channels + 2, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        xx = self.xx.expand(B, -1, -1, -1)   # (B, 1, H, W)
        yy = self.yy.expand(B, -1, -1, -1)   # (B, 1, H, W)
        x = torch.cat([x, xx, yy], dim=1)    # (B, in_channels+2, H, W)
        return self.conv(x)                   # (B, out_channels, H', W')


class BasicBlock(nn.Module):
    """
    ResNet18 BasicBlock with BatchNorm2d.

    Two 3x3 Conv2d layers each followed by BatchNorm2d.
    Shortcut (downsample) applied when spatial dims or channels change.
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


class Actor(nn.Module):
    """
    DDPG Actor network: ResNet18-like CNN with CoordConv stem and BatchNorm backbone.

    Input:  (B, 7, 64, 64) — state s_t = (canvas 3ch, target 3ch, step_normalized 1ch)
    Output: (B, 40) in [0, 1] — k=5 stroke bundles x STROKE_DIM=8 params, via sigmoid

    Architecture:
      CoordConv(7->64, stride=2)  -> (B, 64, 32, 32)
      Stage 1: 2x BasicBlock(64->64)             -> (B, 64,  32, 32)
      Stage 2: 2x BasicBlock(64->128,  stride=2) -> (B, 128, 16, 16)
      Stage 3: 2x BasicBlock(128->256, stride=2) -> (B, 256,  8,  8)
      Stage 4: 2x BasicBlock(256->512, stride=2) -> (B, 512,  4,  4)
      AdaptiveAvgPool2d(1,1) -> flatten           -> (B, 512)
      Linear(512, 40) + sigmoid                   -> (B, 40)

    CoordConv and BasicBlock are shared with models/critic.py (Plan 03-03).
    Uses BatchNorm in backbone (actor.train() during gradient updates,
    actor.eval() during rollout — running stats used at batch=1 inference).
    No weight_norm here — that is the critic's concern.
    """

    def __init__(self):
        super().__init__()
        self.coord_conv = CoordConv(7, 64, kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, STROKES_PER_STEP * STROKE_DIM)

    def _make_layer(self, in_ch: int, out_ch: int, blocks: int,
                    stride: int) -> nn.Sequential:
        """Build a ResNet18 stage as a sequence of BasicBlocks."""
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        layers = [BasicBlock(in_ch, out_ch, stride=stride, downsample=downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 7, 64, 64) -> (B, 40) stroke params in [0, 1]."""
        x = F.relu(self.coord_conv(x), inplace=True)  # (B, 64, 32, 32)
        x = self.layer1(x)                             # (B, 64,  32, 32)
        x = self.layer2(x)                             # (B, 128, 16, 16)
        x = self.layer3(x)                             # (B, 256,  8,  8)
        x = self.layer4(x)                             # (B, 512,  4,  4)
        x = self.pool(x).flatten(1)                    # (B, 512)
        return torch.sigmoid(self.fc(x))               # (B, 40) in [0, 1]
