import torch
import torch.nn.utils.parametrize as P
import pytest
from config import IMG_SIZE
from models.critic import Critic


def test_critic_shape():
    critic = Critic()
    critic.eval()
    with torch.no_grad():
        out = critic(torch.zeros(2, 7, IMG_SIZE, IMG_SIZE))
    assert out.shape == (2, 1), f"Expected (2, 1), got {out.shape}"


def test_critic_unbounded():
    critic = Critic()
    critic.eval()
    with torch.no_grad():
        out = critic(torch.randn(8, 7, IMG_SIZE, IMG_SIZE))
    assert not torch.isnan(out).any(), "Critic output contains NaN"
    # No sigmoid/tanh clamping — value can be any real number (V(s') is unbounded)


def test_critic_no_batchnorm():
    critic = Critic()
    bn = [m for m in critic.modules()
          if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
    assert len(bn) == 0, (
        f"Critic must not use BatchNorm (paper §3.4 — WN+TReLU only). "
        f"Found {len(bn)} BatchNorm modules."
    )


def test_critic_uses_weight_norm():
    critic = Critic()
    wn_convs = [
        m for m in critic.modules()
        if isinstance(m, torch.nn.Conv2d) and P.is_parametrized(m, 'weight')
    ]
    assert len(wn_convs) > 0, (
        "Critic must have at least one Conv2d with weight_norm parametrization "
        "(from torch.nn.utils.parametrizations.weight_norm — deepcopy-safe API)"
    )


def test_critic_input_is_image_not_concat():
    # Confirms the critic consumes a 7ch image through CoordConv,
    # NOT a concatenated (state, action) vector.
    # CoordConv(7, ...) internally creates Conv2d with in_channels=7+2=9.
    critic = Critic()
    assert critic.coord_conv.conv.in_channels == 9, (
        f"Expected critic.coord_conv.conv.in_channels == 9 (7ch image + 2 coord channels), "
        f"got {critic.coord_conv.conv.in_channels}. "
        "This confirms model-based DDPG: critic takes s_{{t+1}} image, not (s,a) concat."
    )


def test_critic_gpu():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    device = torch.device("cuda")
    critic = Critic().to(device)
    critic.eval()
    with torch.no_grad():
        out = critic(torch.zeros(2, 7, IMG_SIZE, IMG_SIZE, device=device))
    assert out.device.type == "cuda"
    assert out.shape == (2, 1)
