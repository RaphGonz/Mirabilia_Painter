import numpy as np
import torch
import pytest
from ddpg.replay_buffer import ReplayBuffer
from config import REPLAY_BUFFER_CAPACITY, IMG_SIZE, STROKES_PER_STEP, STROKE_DIM

# Small capacity used for all functional tests — avoids allocating the full 200k buffer
# (which would consume ~5 GB RAM in CI).
_SMALL_CAP = 100
_ACT_DIM = STROKES_PER_STEP * STROKE_DIM  # 5 * 8 = 40

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_canvas() -> np.ndarray:
    """Return a (6, IMG_SIZE, IMG_SIZE) uint8 zero canvas."""
    return np.zeros((6, IMG_SIZE, IMG_SIZE), dtype=np.uint8)


def _dummy_transition(canvas_val: int = 0) -> dict:
    canvas = np.full((6, IMG_SIZE, IMG_SIZE), canvas_val, dtype=np.uint8)
    return dict(
        obs_canvas=canvas,
        obs_step=float(canvas_val) / 255.0,
        act=np.zeros(_ACT_DIM, dtype=np.float32),
        rew=0.0,
        next_canvas=canvas.copy(),
        next_step=float(canvas_val) / 255.0,
        done=False,
    )


def _push_n(buf: ReplayBuffer, n: int) -> None:
    for i in range(n):
        t = _dummy_transition(i % 256)
        buf.push(
            t["obs_canvas"],
            t["obs_step"],
            t["act"],
            t["rew"],
            t["next_canvas"],
            t["next_step"],
            t["done"],
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_capacity():
    """
    len(buf) grows up to capacity, then wraps (ring buffer semantics).
    - Push 5 < capacity: len == 5
    - Push capacity+10 > capacity: len == capacity (capped), ptr has wrapped
    """
    buf = ReplayBuffer(capacity=_SMALL_CAP)

    # Push fewer than capacity
    _push_n(buf, 5)
    assert len(buf) == 5, f"Expected 5, got {len(buf)}"

    # Push past capacity — size must be capped at capacity
    _push_n(buf, _SMALL_CAP + 10)
    assert len(buf) == _SMALL_CAP, f"Expected {_SMALL_CAP}, got {len(buf)}"
    # ptr wrapped: total pushes = 5 + (capacity+10) = capacity+15, ptr = (capacity+15) % capacity = 15
    expected_ptr = (5 + _SMALL_CAP + 10) % _SMALL_CAP
    assert buf.ptr == expected_ptr, (
        f"ptr should have wrapped to {expected_ptr}, got {buf.ptr}"
    )


def test_uint8_storage():
    """
    Canvas arrays must be stored as uint8 (memory budget DDPG-04 / D-15).
    Step channel must be a 1-D float32 scalar array, NOT tiled to (capacity, 1, 64, 64).
    """
    buf = ReplayBuffer(capacity=_SMALL_CAP)

    # Canvas dtype and shape
    assert buf.obs_canvas.dtype == np.uint8, (
        f"obs_canvas dtype must be np.uint8, got {buf.obs_canvas.dtype}"
    )
    assert buf.obs_canvas.shape == (_SMALL_CAP, 6, IMG_SIZE, IMG_SIZE), (
        f"obs_canvas shape mismatch: {buf.obs_canvas.shape}"
    )
    assert buf.next_canvas.dtype == np.uint8, (
        f"next_canvas dtype must be np.uint8, got {buf.next_canvas.dtype}"
    )
    assert buf.next_canvas.shape == (_SMALL_CAP, 6, IMG_SIZE, IMG_SIZE), (
        f"next_canvas shape mismatch: {buf.next_canvas.shape}"
    )

    # Step dtype and dimensionality — must be 1-D scalar, NOT tiled
    assert buf.obs_step.dtype == np.float32, (
        f"obs_step dtype must be np.float32, got {buf.obs_step.dtype}"
    )
    assert buf.obs_step.ndim == 1, (
        f"obs_step must be 1-D (scalar per transition), got ndim={buf.obs_step.ndim}"
    )
    assert buf.obs_step.shape == (_SMALL_CAP,), (
        f"obs_step shape must be ({_SMALL_CAP},), got {buf.obs_step.shape}"
    )
    assert buf.next_step.dtype == np.float32, (
        f"next_step dtype must be np.float32, got {buf.next_step.dtype}"
    )
    assert buf.next_step.ndim == 1, (
        f"next_step must be 1-D, got ndim={buf.next_step.ndim}"
    )


def test_sample_shapes():
    """
    sample(batch_size, device) must return five float32 tensors with correct shapes.
      obs      : (B, 7, 64, 64)
      act      : (B, 40)
      rew      : (B,)
      next_obs : (B, 7, 64, 64)
      done     : (B,)
    """
    buf = ReplayBuffer(capacity=_SMALL_CAP)
    batch = 16
    _push_n(buf, batch)  # push exactly batch transitions

    device = torch.device("cpu")
    obs, act, rew, next_obs, done = buf.sample(batch, device)

    assert obs.shape == (batch, 7, IMG_SIZE, IMG_SIZE), (
        f"obs shape mismatch: {obs.shape}"
    )
    assert act.shape == (batch, _ACT_DIM), f"act shape mismatch: {act.shape}"
    assert rew.shape == (batch,), f"rew shape mismatch: {rew.shape}"
    assert next_obs.shape == (batch, 7, IMG_SIZE, IMG_SIZE), (
        f"next_obs shape mismatch: {next_obs.shape}"
    )
    assert done.shape == (batch,), f"done shape mismatch: {done.shape}"


def test_sample_dtype_and_range():
    """
    Sampled obs/next_obs must be torch.float32 in [0.0, 1.0] (uint8 / 255 normalisation).
    """
    buf = ReplayBuffer(capacity=_SMALL_CAP)
    _push_n(buf, 32)

    device = torch.device("cpu")
    obs, _, _, next_obs, _ = buf.sample(8, device)

    assert obs.dtype == torch.float32, f"obs dtype must be float32, got {obs.dtype}"
    assert next_obs.dtype == torch.float32, (
        f"next_obs dtype must be float32, got {next_obs.dtype}"
    )
    assert obs.max() <= 1.0, f"obs.max() = {obs.max()} exceeds 1.0"
    assert obs.min() >= 0.0, f"obs.min() = {obs.min()} is negative"
    assert next_obs.max() <= 1.0, f"next_obs.max() = {next_obs.max()} exceeds 1.0"
    # Validate canvas and step channels independently:
    # step channel stored as normalised scalar in [0,1] — a raw integer step
    # index (0-40) would exceed 1.0 but would not be caught by aggregate max.
    canvas_part = obs[:, :6, :, :]   # 3ch current + 3ch target (uint8/255)
    step_part   = obs[:, 6:7, :, :]  # normalised step in [0, 1]
    assert canvas_part.max() <= 1.0, f"canvas channels exceed 1.0: {canvas_part.max()}"
    assert canvas_part.min() >= 0.0, f"canvas channels below 0.0: {canvas_part.min()}"
    assert step_part.max() <= 1.0,   f"step channel exceeds 1.0: {step_part.max()}"
    assert step_part.min() >= 0.0,   f"step channel below 0.0: {step_part.min()}"


def test_sample_gpu():
    """Sampled tensors must reside on the requested CUDA device."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")

    buf = ReplayBuffer(capacity=_SMALL_CAP)
    _push_n(buf, 16)

    device = torch.device("cuda")
    obs, act, rew, next_obs, done = buf.sample(8, device)

    assert obs.device.type == "cuda", f"obs not on CUDA: {obs.device}"
    assert next_obs.device.type == "cuda", f"next_obs not on CUDA: {next_obs.device}"
    assert act.device.type == "cuda", f"act not on CUDA: {act.device}"
