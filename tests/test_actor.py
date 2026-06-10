import torch
import pytest
from config import IMG_SIZE, STROKE_DIM, STROKES_PER_STEP
from models.actor import Actor, CoordConv


def test_actor_shape():
    actor = Actor()
    actor.eval()
    with torch.no_grad():
        out = actor(torch.zeros(2, 7, IMG_SIZE, IMG_SIZE))
    assert out.shape == (2, STROKES_PER_STEP * STROKE_DIM)


def test_output_range():
    actor = Actor()
    actor.eval()
    with torch.no_grad():
        out = actor(torch.rand(8, 7, IMG_SIZE, IMG_SIZE))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_actor_has_batchnorm():
    actor = Actor()
    bn_modules = [m for m in actor.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert len(bn_modules) > 0


def test_coordconv_appends_two_channels():
    actor = Actor()
    # CoordConv(in_channels=7, ...) must create Conv2d with 7+2=9 input channels
    assert actor.coord_conv.conv.in_channels == 9


def test_actor_gpu():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    device = torch.device("cuda")
    actor = Actor().to(device)
    actor.eval()
    with torch.no_grad():
        out = actor(torch.zeros(1, 7, IMG_SIZE, IMG_SIZE, device=device))
    assert out.device.type == "cuda"
    assert out.shape == (1, STROKES_PER_STEP * STROKE_DIM)
