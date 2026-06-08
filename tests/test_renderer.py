import torch
import pytest
from renderer import draw


def test_draw_output_shape():
    canvas = torch.zeros(3, 64, 64)
    params = torch.tensor([0.5, 0.5, 0.3, 0.1, 0.0, 1.0, 0.0, 0.0])
    result = draw(canvas, params)
    assert result.shape == (3, 64, 64)


def test_draw_no_autograd():
    canvas = torch.zeros(3, 64, 64)
    params = torch.tensor([0.5, 0.5, 0.3, 0.1, 0.0, 1.0, 0.0, 0.0])
    result = draw(canvas, params)
    assert not result.requires_grad


def test_draw_paints_pixels():
    canvas = torch.zeros(3, 64, 64)
    params = torch.tensor([0.5, 0.5, 0.3, 0.2, 0.0, 1.0, 0.0, 0.0])
    result = draw(canvas, params)
    assert (result > 0).any()


def test_draw_full_canvas():
    canvas = torch.zeros(3, 64, 64)
    params = torch.tensor([0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    result = draw(canvas, params)
    assert (result[1] > 0).sum().item() == 64 * 64


def test_draw_subpixel_stroke_is_empty():
    """Stroke smaller than a pixel — correct behavior is unmodified canvas."""
    canvas = torch.zeros(3, 64, 64)
    params = torch.tensor([0.5, 0.5, 0.001, 0.001, 0.0, 1.0, 0.0, 0.0])
    result = draw(canvas, params)
    assert torch.equal(result, canvas)


def test_draw_values_in_range():
    canvas = torch.zeros(3, 64, 64)
    params = torch.tensor([0.3, 0.7, 0.2, 0.1, 0.25, 0.8, 0.5, 0.2])
    result = draw(canvas, params)
    assert result.min() >= 0.0 and result.max() <= 1.0


def test_draw_extreme_rotations():
    """Verify draw works at theta boundaries: 0.0, 0.5 (90 deg), 1.0 (180 deg)."""
    canvas = torch.zeros(3, 64, 64)
    for theta in [0.0, 0.5, 1.0]:
        params = torch.tensor([0.5, 0.5, 0.3, 0.1, theta, 1.0, 0.0, 0.0])
        result = draw(canvas, params)
        assert result.shape == (3, 64, 64)
        assert result.min() >= 0.0 and result.max() <= 1.0


def test_draw_gpu():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    device = torch.device("cuda")
    canvas = torch.zeros(3, 64, 64, device=device)
    params = torch.tensor([0.5, 0.5, 0.3, 0.2, 0.0, 1.0, 0.0, 0.0], device=device)
    result = draw(canvas, params)
    assert result.device.type == "cuda"
