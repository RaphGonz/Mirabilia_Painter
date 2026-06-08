# No imports. No logic. Constants only.

IMG_SIZE: int = 64
STROKE_DIM: int = 8           # (cx, cy, w, h, theta, r, g, b)
STROKES_PER_STEP: int = 5
N_STROKES: int = 40           # steps per episode; 40 steps x 5 strokes = 200 total
IMAGE_RANGE: tuple = (0.0, 1.0)

# Default colorspace for palette projection
PALETTE_COLORSPACE: str = "rgb"   # options: "rgb", "oklab", "hsv"

# MIN_VISIBLE_STROKE_WIDTH = 2.0 / (IMG_SIZE - 1)  # ~0.032 at 64x64 (comment only, not a constant)
