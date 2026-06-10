import torch
import pytest
from config import IMG_SIZE, STROKE_DIM
from models.renderer import SoftRasterizer, NeuralRenderer


def test_output_shape():
    R = SoftRasterizer()
    out = R(torch.rand(4, STROKE_DIM))
    assert out.shape == (4, 3, IMG_SIZE, IMG_SIZE)


def test_output_range():
    R = SoftRasterizer()
    out = R(torch.rand(32, STROKE_DIM))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_single_sample():
    R = SoftRasterizer()
    R.eval()
    with torch.no_grad():
        out = R(torch.rand(1, STROKE_DIM))
    assert out.shape == (1, 3, IMG_SIZE, IMG_SIZE)


def test_compute_alpha_shape():
    R = SoftRasterizer()
    alpha = R.compute_alpha(torch.rand(4, STROKE_DIM))
    assert alpha.shape == (4, 1, IMG_SIZE, IMG_SIZE)
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0


def test_premultiplied_consistency():
    """forward() = compute_alpha() * color, element-wise."""
    R = SoftRasterizer()
    params = torch.rand(4, STROKE_DIM)
    with torch.no_grad():
        premul = R(params)
        alpha  = R.compute_alpha(params)
        color  = params[:, 5:8].view(4, 3, 1, 1)
    assert torch.allclose(premul, alpha * color, atol=1e-6)


def test_geometry_thin_vs_wide():
    """Thin stroke must have smaller total alpha than wide stroke."""
    R = SoftRasterizer(beta=0.5)
    thin = torch.tensor([[0.5, 0.5, 0.3, 0.02, 0.0, 1.0, 0.0, 0.0]])
    wide = torch.tensor([[0.5, 0.5, 0.3, 0.5,  0.0, 1.0, 0.0, 0.0]])
    with torch.no_grad():
        a_thin = R.compute_alpha(thin).sum()
        a_wide = R.compute_alpha(wide).sum()
    assert a_thin < a_wide, "Thin stroke must have less total alpha than wide"


def test_geometry_tilted_vs_straight():
    """Tilted and straight strokes must look different (different alpha distributions)."""
    R = SoftRasterizer(beta=0.5)
    straight = torch.tensor([[0.5, 0.5, 0.4, 0.1, 0.0,  1.0, 0.0, 0.0]])
    tilted   = torch.tensor([[0.5, 0.5, 0.4, 0.1, 0.45, 1.0, 0.0, 0.0]])
    with torch.no_grad():
        a_s = R.compute_alpha(straight)[0, 0]
        a_t = R.compute_alpha(tilted)[0, 0]
    # Alpha distributions must differ meaningfully
    diff = (a_s - a_t).abs().mean().item()
    assert diff > 0.01, f"Tilted/straight nearly identical: mean diff = {diff:.4f}"


def test_no_batchnorm():
    R = SoftRasterizer()
    bn = [m for m in R.modules()
          if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
    assert len(bn) == 0


def test_gpu():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    device = torch.device("cuda")
    R = SoftRasterizer().to(device)
    out = R(torch.rand(1, STROKE_DIM, device=device))
    assert out.device.type == "cuda"
    assert out.shape == (1, 3, IMG_SIZE, IMG_SIZE)


def test_freeze_assertion():
    """Param norm must not change after frozen forward pass (REND-03)."""
    R = SoftRasterizer()
    R.eval()
    for p in R.parameters():
        p.requires_grad_(False)

    def param_norm(m):
        return sum(p.data.norm(2).item() ** 2 for p in m.parameters()) ** 0.5

    norm_before = param_norm(R)
    with torch.no_grad():
        _ = R(torch.rand(4, STROKE_DIM))
    assert abs(param_norm(R) - norm_before) < 1e-6


def test_backward_compat_alias():
    """NeuralRenderer alias must point to SoftRasterizer."""
    assert NeuralRenderer is SoftRasterizer
