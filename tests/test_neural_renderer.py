# Source: CONTEXT.md D-09/D-10/D-11 + REND-01/REND-03 (2026-06-09)
import torch
import pytest
from config import IMG_SIZE, STROKE_DIM
from models.renderer import NeuralRenderer


def test_neural_renderer_output_shape():
    R = NeuralRenderer()
    x = torch.rand(4, STROKE_DIM)
    out = R(x)
    assert out.shape == (4, 3, IMG_SIZE, IMG_SIZE), f"Expected (4,3,{IMG_SIZE},{IMG_SIZE}), got {out.shape}"


def test_neural_renderer_output_range():
    R = NeuralRenderer()
    out = R(torch.rand(4, STROKE_DIM))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_neural_renderer_no_batchnorm():
    R = NeuralRenderer()
    bn_layers = [m for m in R.modules()
                 if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
    assert len(bn_layers) == 0, f"Found BatchNorm layers: {bn_layers}"


def test_neural_renderer_single_sample():
    R = NeuralRenderer()
    R.eval()
    for p in R.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        out = R(torch.rand(1, STROKE_DIM))
    assert out.shape == (1, 3, IMG_SIZE, IMG_SIZE)
    assert not out.requires_grad


def test_neural_renderer_gpu():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    device = torch.device("cuda")
    R = NeuralRenderer().to(device)
    out = R(torch.rand(1, STROKE_DIM, device=device))
    assert out.device.type == "cuda"
    assert out.shape == (1, 3, IMG_SIZE, IMG_SIZE)


def test_freeze_assertion():
    """Param norm must not change after frozen forward pass (REND-03)."""
    R = NeuralRenderer()
    # Simulate load_frozen_renderer without file I/O
    R.eval()
    for p in R.parameters():
        p.requires_grad_(False)

    def param_norm(m):
        return sum(p.data.norm(2).item() ** 2 for p in m.parameters()) ** 0.5

    norm_before = param_norm(R)
    with torch.no_grad():
        _ = R(torch.rand(4, STROKE_DIM))
    assert abs(param_norm(R) - norm_before) < 1e-6
