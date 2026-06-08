import pytest
from config import IMG_SIZE, STROKE_DIM, STROKES_PER_STEP, N_STROKES, IMAGE_RANGE, PALETTE_COLORSPACE


def test_img_size():
    assert IMG_SIZE == 64


def test_stroke_dim():
    assert STROKE_DIM == 8


def test_strokes_per_step():
    assert STROKES_PER_STEP == 5


def test_n_strokes():
    assert N_STROKES == 40


def test_image_range():
    assert IMAGE_RANGE == (0.0, 1.0)


def test_palette_colorspace_valid():
    assert PALETTE_COLORSPACE in {"rgb", "oklab", "hsv"}


def test_all_constants_types():
    assert isinstance(IMG_SIZE, int)
    assert isinstance(STROKE_DIM, int)
    assert isinstance(STROKES_PER_STEP, int)
    assert isinstance(N_STROKES, int)
    assert isinstance(IMAGE_RANGE, tuple)
    assert isinstance(PALETTE_COLORSPACE, str)
