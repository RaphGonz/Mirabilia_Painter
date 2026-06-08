import torch
import math
from config import PALETTE_COLORSPACE

# Physical paint palette — edit these values after measuring with paint mixer.
# Values are sRGB float in [0.0, 1.0]: divide uint8 (0-255) values by 255.
# Placeholder — user must replace with actual ~40 colors from paint mixer.
_PALETTE_SRGB: list[tuple[float, float, float]] = [
    (1.000, 1.000, 1.000),  # white
    (0.000, 0.000, 0.000),  # black
    (1.000, 0.000, 0.000),  # red
    (0.000, 1.000, 0.000),  # green
    (0.000, 0.000, 1.000),  # blue
    (1.000, 1.000, 0.000),  # yellow
    # TODO: Replace with actual ~40 colors from physical paint mixer.
    # Divide uint8 (0-255) values by 255.0 before entering here.
]

# Shape (P, 3), float32 — created once at module load, not per-call
PALETTE: torch.Tensor = torch.tensor(_PALETTE_SRGB, dtype=torch.float32)


def _srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    """Apply inverse sRGB gamma to get linear light values. [VERIFIED: sRGB IEC 61966-2-1 spec]"""
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_oklab(rgb_linear: torch.Tensor) -> torch.Tensor:
    """
    Convert linear sRGB to oklab.
    rgb_linear: (..., 3) in [0, 1] linear light.
    Returns: (..., 3) [L, a, b]
    Source: https://bottosson.github.io/posts/oklab/
    """
    r, g, b = rgb_linear[..., 0], rgb_linear[..., 1], rgb_linear[..., 2]
    l = 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
    m = 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
    s = 0.0883024619*r + 0.2817188376*g + 0.6299787005*b
    # clamp before pow(1/3) — LMS can be tiny-negative due to float32 error near black
    l_ = l.clamp(min=0.0).pow(1.0 / 3.0)
    m_ = m.clamp(min=0.0).pow(1.0 / 3.0)
    s_ = s.clamp(min=0.0).pow(1.0 / 3.0)
    L  =  0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_
    a  =  1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_
    b_ =  0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_
    return torch.stack([L, a, b_], dim=-1)


def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """Convert sRGB to HSV. rgb: (..., 3) in [0, 1]. Returns (..., 3) [H, S, V]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = torch.max(rgb, dim=-1).values
    cmin = torch.min(rgb, dim=-1).values
    delta = cmax - cmin
    v = cmax
    s = torch.where(cmax > 0, delta / cmax, torch.zeros_like(cmax))
    h = torch.zeros_like(cmax)
    h = torch.where((cmax == r) & (delta > 0), ((g - b) / delta) % 6.0, h)
    h = torch.where((cmax == g) & (delta > 0), (b - r) / delta + 2.0, h)
    h = torch.where((cmax == b) & (delta > 0), (r - g) / delta + 4.0, h)
    h = h / 6.0
    return torch.stack([h, s, v], dim=-1)


def project_color(
    rgb: tuple | list | torch.Tensor,
    colorspace: str = PALETTE_COLORSPACE,
) -> torch.Tensor:
    """
    Return the nearest palette color to `rgb` in the given colorspace.

    Args:
        rgb: sRGB color, shape (3,) or length-3 sequence, values in [0, 1]
        colorspace: one of "rgb", "oklab", "hsv"

    Returns:
        Nearest palette color as tensor (3,) in sRGB [0, 1]

    Raises:
        ValueError: if colorspace is not one of "rgb", "oklab", "hsv"
    """
    q = torch.as_tensor(rgb, dtype=torch.float32).unsqueeze(0)  # (1, 3) — cdist requires 2D
    pal = PALETTE.float()

    if colorspace == "oklab":
        q_conv = _linear_to_oklab(_srgb_to_linear(q))
        pal_conv = _linear_to_oklab(_srgb_to_linear(pal))
    elif colorspace == "hsv":
        q_conv = _rgb_to_hsv(q)
        pal_conv = _rgb_to_hsv(pal)
    elif colorspace == "rgb":
        q_conv = q
        pal_conv = pal
    else:
        raise ValueError(
            f"Unsupported colorspace: {colorspace!r}. Must be one of 'rgb', 'oklab', 'hsv'."
        )

    dists = torch.cdist(q_conv, pal_conv)   # (1, P)
    idx = dists.argmin(dim=-1).item()
    return PALETTE[idx]
