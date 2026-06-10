# No imports. No logic. Constants only.

IMG_SIZE: int = 64
STROKE_DIM: int = 8           # (cx, cy, w, h, theta, r, g, b)
STROKES_PER_STEP: int = 5
N_STROKES: int = 40           # steps per episode; 40 steps x 5 strokes = 200 total
IMAGE_RANGE: tuple = (0.0, 1.0)

# Default colorspace for palette projection
PALETTE_COLORSPACE: str = "rgb"   # options: "rgb", "oklab", "hsv"

# SoftRasterizer edge softness — controls sigmoid transition width in pixels
# 0.5 → ~2px (sharp edges, steeper gradients)
# 1.0 → ~4px (recommended baseline)
# 2.0 → ~9px (very soft, smooth gradients but blurry strokes)
RENDERER_BETA: float = 1.0

# MIN_VISIBLE_STROKE_WIDTH = 2.0 / (IMG_SIZE - 1)  # ~0.032 at 64x64 (comment only, not a constant)

# ---------------------------------------------------------------------------
# DDPG hyperparameters — from paper Table 1 (LearningToPaint appendix §7.1)
# These are the starting values; autoresearch may vary them.
# ---------------------------------------------------------------------------
ACTOR_LR: float = 3e-4          # decays to ACTOR_LR_FINAL after LR_DECAY_STEP batches
ACTOR_LR_FINAL: float = 1e-4
CRITIC_LR: float = 1e-3         # decays to CRITIC_LR_FINAL after LR_DECAY_STEP batches
CRITIC_LR_FINAL: float = 3e-4
LR_DECAY_STEP: int = 100_000    # number of training batches before LR decay

GAMMA: float = 0.955            # γ^k where k=5 (action bundle); effective per-step discount
TAU: float = 0.005              # soft target update rate: θ_target ← τ·θ + (1-τ)·θ_target
BATCH_SIZE: int = 96            # training minibatch size (paper Table 1)
REPLAY_BUFFER_CAPACITY: int = 200_000  # transitions; larger than paper (800 ep) for MS COCO diversity
GRAD_CLIP_CRITIC: float = 1.0   # max_norm for critic gradient clipping
