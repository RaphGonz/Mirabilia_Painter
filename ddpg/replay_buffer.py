import numpy as np
import torch

from config import IMG_SIZE, STROKES_PER_STEP, STROKE_DIM, REPLAY_BUFFER_CAPACITY


class ReplayBuffer:
    """
    Off-policy experience replay buffer for DDPG.

    Storage layout (memory budget — DDPG-04 / D-15):
      - Canvas channels (3ch current + 3ch target) stored as uint8 [0, 255].
        Shape: (capacity, 6, IMG_SIZE, IMG_SIZE).
        At 200k transitions: ~6 × 200k × 64 × 64 = ~4.9 GB (uint8).
      - Step channel stored as a float32 SCALAR per transition (NOT tiled).
        Shape: (capacity,).
        Tiling to (1, H, W) happens lazily in sample() to save ~3.3 GB.
      - Full buffer at capacity=200k: ~9.84 GB total (obs+next_obs combined),
        fitting within 31 GB RAM.

    On-sample conversion (D-17):
      canvas uint8 → float32 [0, 1]  via  .float().div(255.0)
      step scalar  → (B, 1, H, W)    via  .view(-1, 1, 1, 1).expand(-1, 1, H, W)
      Both are concatenated to produce a 7-channel observation tensor.

    Push signature:
      push(obs_canvas, obs_step, act, rew, next_canvas, next_step, done)
        obs_canvas  : ndarray (6, IMG_SIZE, IMG_SIZE) uint8
        obs_step    : float  — normalised step index t/N_STROKES in [0, 1]
        act         : ndarray (STROKES_PER_STEP * STROKE_DIM,) float32
        rew         : float
        next_canvas : ndarray (6, IMG_SIZE, IMG_SIZE) uint8
        next_step   : float
        done        : bool

    Sample returns (obs, act, rew, next_obs, done) — all torch.float32 on device.
    """

    def __init__(self, capacity: int = REPLAY_BUFFER_CAPACITY) -> None:
        self.capacity = capacity
        self.ptr: int = 0
        self.size: int = 0

        # 6-channel canvas stored as uint8 (3ch current + 3ch target) — D-15/D-16
        self.obs_canvas = np.zeros(
            (capacity, 6, IMG_SIZE, IMG_SIZE), dtype=np.uint8
        )
        self.next_canvas = np.zeros(
            (capacity, 6, IMG_SIZE, IMG_SIZE), dtype=np.uint8
        )

        # Step encoded as a scalar float32 per transition — NOT tiled (saves 3.3 GB) — D-15
        self.obs_step = np.zeros((capacity,), dtype=np.float32)
        self.next_step = np.zeros((capacity,), dtype=np.float32)

        # Action, reward, done
        self.actions = np.zeros(
            (capacity, STROKES_PER_STEP * STROKE_DIM), dtype=np.float32
        )
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=bool)

    # ------------------------------------------------------------------
    # Push
    # ------------------------------------------------------------------

    def push(
        self,
        obs_canvas: np.ndarray,
        obs_step: float,
        act: np.ndarray,
        rew: float,
        next_canvas: np.ndarray,
        next_step: float,
        done: bool,
    ) -> None:
        """Insert one transition; wraps around when full (O(1) ring insert)."""
        idx = self.ptr
        self.obs_canvas[idx] = obs_canvas      # (6, H, W) uint8
        self.obs_step[idx] = obs_step          # scalar float32
        self.actions[idx] = act               # (40,) float32
        self.rewards[idx] = rew
        self.next_canvas[idx] = next_canvas   # (6, H, W) uint8
        self.next_step[idx] = next_step       # scalar float32
        self.dones[idx] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------

    def sample(self, batch_size: int, device: torch.device):
        """
        Draw a random mini-batch of transitions.

        Returns:
            obs      : (B, 7, H, W) float32 — canvas [0,1] + tiled step channel
            act      : (B, STROKES_PER_STEP * STROKE_DIM) float32
            rew      : (B,) float32
            next_obs : (B, 7, H, W) float32
            done     : (B,) bool
        """
        idx = np.random.randint(0, self.size, size=batch_size)

        # --- Current obs ---
        # uint8 canvas → float32 [0, 1]
        canvas = torch.from_numpy(self.obs_canvas[idx]).float().div(255.0).to(device)
        H, W = canvas.shape[-2], canvas.shape[-1]
        step = torch.from_numpy(self.obs_step[idx]).to(device)
        # Tile scalar step to (B, 1, H, W) — done here only, NOT stored
        step_ch = step.view(-1, 1, 1, 1).expand(-1, 1, H, W)
        obs = torch.cat([canvas, step_ch], dim=1)  # (B, 7, H, W)

        # --- Next obs ---
        n_canvas = torch.from_numpy(self.next_canvas[idx]).float().div(255.0).to(device)
        n_step = torch.from_numpy(self.next_step[idx]).to(device)
        n_step_ch = n_step.view(-1, 1, 1, 1).expand(-1, 1, H, W)
        next_obs = torch.cat([n_canvas, n_step_ch], dim=1)  # (B, 7, H, W)

        # --- Action / reward / done ---
        act = torch.from_numpy(self.actions[idx]).to(device)
        rew = torch.from_numpy(self.rewards[idx]).to(device)
        done = torch.from_numpy(self.dones[idx]).to(device)

        return obs, act, rew, next_obs, done

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.size
