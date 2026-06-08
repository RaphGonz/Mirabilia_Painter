import torch
import pytest
from palette import project_color, PALETTE


def test_project_color_returns_tensor():
    result = project_color((1.0, 0.0, 0.0))
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3,)


def test_project_color_in_palette():
    """Result must be one of the palette entries."""
    result = project_color((0.5, 0.5, 0.5))
    matches = (PALETTE == result.unsqueeze(0)).all(dim=1)
    assert matches.any()


def test_project_color_rgb():
    result = project_color((1.0, 1.0, 1.0), colorspace="rgb")
    assert result.shape == (3,)


def test_project_color_oklab():
    result = project_color((1.0, 1.0, 1.0), colorspace="oklab")
    assert result.shape == (3,)


def test_project_color_hsv():
    result = project_color((1.0, 1.0, 1.0), colorspace="hsv")
    assert result.shape == (3,)


def test_project_color_invalid_colorspace():
    with pytest.raises((ValueError, KeyError, AssertionError)):
        project_color((0.5, 0.5, 0.5), colorspace="xyz")


def test_project_color_black():
    """Black input — oklab NaN guard must hold."""
    result = project_color((0.0, 0.0, 0.0), colorspace="oklab")
    assert result.shape == (3,)
    assert not result.isnan().any()


def test_project_color_accepts_tensor():
    rgb = torch.tensor([0.5, 0.3, 0.1])
    result = project_color(rgb)
    assert result.shape == (3,)
