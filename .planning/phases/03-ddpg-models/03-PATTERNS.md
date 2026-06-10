# Phase 3: DDPG Models — Pattern Map

**Mapped:** 2026-06-10
**Files analyzed:** 7 (4 implementation + 1 init + 4 test files per research plan, grouped as 6 deliverables below)
**Analogs found:** 6 / 6

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `models/actor.py` | model | request-response | `models/renderer.py` | role-match (same nn.Module conventions, same package) |
| `models/critic.py` | model | request-response | `models/renderer.py` | role-match (same nn.Module conventions) |
| `ddpg/agent.py` | service | request-response | `pretrain_renderer.py` (freeze/load helpers) | partial (same torch patterns, no exact analog) |
| `ddpg/replay_buffer.py` | utility | batch | `pretrain_renderer.py` (numpy array ops) | partial (no ring-buffer analog exists) |
| `ddpg/__init__.py` | config | — | `models/__init__.py` (empty) | exact |
| `tests/test_actor.py`, `tests/test_critic.py`, `tests/test_agent.py`, `tests/test_replay_buffer.py` | test | — | `tests/test_neural_renderer.py` | exact |

---

## Pattern Assignments

### `models/actor.py` (model, request-response)

**Analog:** `models/renderer.py`

**Imports pattern** (`models/renderer.py` lines 1–4):
```python
import torch
import torch.nn as nn
from config import IMG_SIZE, STROKE_DIM, STROKES_PER_STEP, N_STROKES
```
Note: Actor also needs `torch.nn.functional as F` and `import math` is not needed (no pixel grids).
Config constants to import: `IMG_SIZE`, `STROKES_PER_STEP`, `STROKE_DIM`.
Output size: `STROKES_PER_STEP * STROKE_DIM = 40`.

**register_buffer pattern** (`models/renderer.py` lines 33–37):
```python
# Pixel coordinate grids — buffers move to device with .to(device)
y = torch.arange(IMG_SIZE, dtype=torch.float32)
x = torch.arange(IMG_SIZE, dtype=torch.float32)
yy, xx = torch.meshgrid(y, x, indexing='ij')
self.register_buffer('xx', xx.unsqueeze(0).contiguous())  # (1, H, W)
self.register_buffer('yy', yy.unsqueeze(0).contiguous())  # (1, H, W)
```
For CoordConv: use `register_buffer` for the normalized x/y coordinate grids so they move to device with `.to(device)`. Pattern is identical but with `torch.linspace(-1, 1, ...)` instead of `torch.arange`.

**Class structure and docstring pattern** (`models/renderer.py` lines 7–26):
```python
class SoftRasterizer(nn.Module):
    """
    [one-line description]

    Input:  (B, STROKE_DIM=8) ...
    Output: (B, 3, IMG_SIZE, IMG_SIZE) ...

    [behaviour notes]
    """

    def __init__(self, ...):
        super().__init__()
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, IN) → (B, OUT) [description]."""
        ...
```
Copy this docstring style exactly. Actor docstring should document: Input `(B, 7, 64, 64)`, Output `(B, 40)` in `[0, 1]`.

**Core architecture pattern** (from RESEARCH.md Pattern 1 + Pattern 2):
```python
class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        H = W = IMG_SIZE
        xx = torch.linspace(-1, 1, W).view(1, 1, 1, W).expand(1, 1, H, W)
        yy = torch.linspace(-1, 1, H).view(1, 1, H, 1).expand(1, 1, H, W)
        self.register_buffer('xx', xx.contiguous())
        self.register_buffer('yy', yy.contiguous())
        self.conv = nn.Conv2d(in_channels + 2, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)

    def forward(self, x):
        B = x.shape[0]
        xx = self.xx.expand(B, -1, -1, -1)
        yy = self.yy.expand(B, -1, -1, -1)
        x = torch.cat([x, xx, yy], dim=1)
        return self.conv(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.coord_conv = CoordConv(7, 64, kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d((1, 1))
        self.fc     = nn.Linear(512, STROKES_PER_STEP * STROKE_DIM)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        layers = [BasicBlock(in_ch, out_ch, stride=stride, downsample=downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.coord_conv(x), inplace=True)  # (B, 64, 32, 32)
        x = self.layer1(x)   # (B, 64, 32, 32)
        x = self.layer2(x)   # (B, 128, 16, 16)
        x = self.layer3(x)   # (B, 256, 8, 8)
        x = self.layer4(x)   # (B, 512, 4, 4)
        x = self.pool(x).flatten(1)  # (B, 512)
        return torch.sigmoid(self.fc(x))  # (B, 40)
```

---

### `models/critic.py` (model, request-response)

**Analog:** `models/renderer.py` (same package, same nn.Module structure)

**Imports pattern** (`models/renderer.py` lines 1–4, adapted):
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from config import IMG_SIZE, STROKES_PER_STEP, STROKE_DIM
```
Critical: use `from torch.nn.utils.parametrizations import weight_norm` — NOT `torch.nn.utils.weight_norm` (deprecated, deepcopy bug per RESEARCH.md Pitfall 1).

**Core WN+TReLU pattern** (from RESEARCH.md Pattern 3):
```python
class TReLU(nn.Module):
    """Translated ReLU: F.relu(x - alpha) + alpha, scalar alpha per instance."""
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return F.relu(x - self.alpha) + self.alpha  # NOT inplace — corrupts graph


class BasicBlockWN(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv2d(in_channels, out_channels, 3,
                                           stride=stride, padding=1, bias=False))
        self.relu1 = TReLU()
        self.conv2 = weight_norm(nn.Conv2d(out_channels, out_channels, 3,
                                           stride=1, padding=1, bias=False))
        self.relu2 = TReLU()
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu2(out + identity)
```

**Critic head** (from RESEARCH.md code examples):
```python
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        # CoordConv reused from actor.py (shared helper module)
        self.coord_conv = CoordConv(7, 64, kernel_size=3, stride=2, padding=1)
        # Same 4 stages as actor but using BasicBlockWN
        ...
        self.fc = nn.Linear(512, 1)  # No activation — unbounded V(s')

    def forward(self, x):
        ...
        return self.fc(x)  # (B, 1) — no sigmoid, no tanh
```
The critic's `_make_layer` downsample convolution should also be wrapped with `weight_norm` (A4 in RESEARCH.md assumptions).

---

### `ddpg/agent.py` (service, request-response)

**Analog:** `pretrain_renderer.py` (load_frozen_renderer pattern, lines 57–70)

**Freeze/deepcopy pattern** (`pretrain_renderer.py` lines 57–70):
```python
def load_frozen_renderer(path: str, device: torch.device) -> SoftRasterizer:
    R = SoftRasterizer()
    R.load_state_dict(torch.load(path, weights_only=True))
    R.eval()
    for p in R.parameters():
        p.requires_grad_(False)
    return R.to(device)
```
Copy this `eval()` + `requires_grad_(False)` double-freeze pattern for target network init in `agent.py`. Target networks are deepcopy'd (not loaded from disk), but the freeze sequence is identical.

**Target network init pattern** (from RESEARCH.md Pattern 6):
```python
import copy
from models.actor import Actor
from models.critic import Critic
from config import TAU, GAMMA, ACTOR_LR, CRITIC_LR, GRAD_CLIP_CRITIC

class DDPGAgent:
    def __init__(self, device):
        self.device = device
        self.actor  = Actor().to(device)
        self.critic = Critic().to(device)

        # deepcopy-safe because critic uses parametrizations.weight_norm
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        # Permanently freeze targets
        self.actor_target.eval()
        self.critic_target.eval()
        for p in self.actor_target.parameters():
            p.requires_grad_(False)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=ACTOR_LR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
```

**Soft update pattern** (from RESEARCH.md Pattern 4):
```python
@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for p_targ, p in zip(target.parameters(), source.parameters()):
        p_targ.data.mul_(1.0 - tau)
        p_targ.data.add_(tau * p.data)
```

**update_step scaffold** (Phase 3 provides structure; SoftRasterizer integration is a Phase 4 TODO):
```python
def update_step(self, batch):
    obs, act, rew, next_obs, done = batch
    # --- Critic update ---
    with torch.no_grad():
        v_next = self.critic_target(next_obs)         # (B, 1)
        y = rew.unsqueeze(1) + GAMMA * v_next * (~done.unsqueeze(1))
    v_pred = self.critic(next_obs)                    # (B, 1)  NOTE: critic takes s_{t+1}
    critic_loss = F.mse_loss(v_pred, y)
    self.critic_opt.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP_CRITIC)
    self.critic_opt.step()
    # --- Actor update ---
    # TODO Phase 4: actor_loss = -self.critic(render_next_state(self.actor(obs))).mean()
    # --- Soft updates ---
    soft_update(self.actor_target,  self.actor,  TAU)
    soft_update(self.critic_target, self.critic, TAU)
    return critic_loss.item()
```

---

### `ddpg/replay_buffer.py` (utility, batch)

**Analog:** No direct analog in codebase. Pattern from RESEARCH.md Pattern 5.

**Imports** (follow `models/renderer.py` import style, no relative imports):
```python
import numpy as np
import torch
from config import IMG_SIZE, STROKES_PER_STEP, STROKE_DIM, REPLAY_BUFFER_CAPACITY
```

**Core ring buffer pattern** (from RESEARCH.md Pattern 5):
```python
class ReplayBuffer:
    def __init__(self, capacity: int = REPLAY_BUFFER_CAPACITY):
        self.capacity = capacity
        self.ptr  = 0
        self.size = 0
        # 6ch canvas stored as uint8 (canvas 3ch + target 3ch) — D-15/D-16
        self.obs_canvas  = np.zeros((capacity, 6, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        self.next_canvas = np.zeros((capacity, 6, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        # step scalar stored as float32 scalar, NOT tiled (saves 3.3 GB) — D-15
        self.obs_step    = np.zeros((capacity,), dtype=np.float32)
        self.next_step   = np.zeros((capacity,), dtype=np.float32)
        self.actions     = np.zeros((capacity, STROKES_PER_STEP * STROKE_DIM), dtype=np.float32)
        self.rewards     = np.zeros((capacity,), dtype=np.float32)
        self.dones       = np.zeros((capacity,), dtype=bool)

    def push(self, obs_canvas, obs_step, act, rew, next_canvas, next_step, done):
        idx = self.ptr
        self.obs_canvas[idx]  = obs_canvas    # uint8 ndarray (6, H, W)
        self.obs_step[idx]    = obs_step      # float32 scalar
        self.actions[idx]     = act
        self.rewards[idx]     = rew
        self.next_canvas[idx] = next_canvas
        self.next_step[idx]   = next_step
        self.dones[idx]       = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        # Convert uint8 canvas to float32 [0,1] — D-17
        canvas  = torch.from_numpy(self.obs_canvas[idx]).float().div(255.0).to(device)
        step    = torch.from_numpy(self.obs_step[idx]).to(device)
        H, W    = canvas.shape[-2], canvas.shape[-1]
        step_ch = step.view(-1, 1, 1, 1).expand(-1, 1, H, W)
        obs     = torch.cat([canvas, step_ch], dim=1)   # (B, 7, 64, 64)

        n_canvas = torch.from_numpy(self.next_canvas[idx]).float().div(255.0).to(device)
        n_step   = torch.from_numpy(self.next_step[idx]).to(device)
        n_step_ch = n_step.view(-1, 1, 1, 1).expand(-1, 1, H, W)
        next_obs  = torch.cat([n_canvas, n_step_ch], dim=1)  # (B, 7, 64, 64)

        act  = torch.from_numpy(self.actions[idx]).to(device)
        rew  = torch.from_numpy(self.rewards[idx]).to(device)
        done = torch.from_numpy(self.dones[idx]).to(device)
        return obs, act, rew, next_obs, done

    def __len__(self):
        return self.size
```

---

### `ddpg/__init__.py` (config, —)

**Analog:** `models/__init__.py` (exact match — empty file)

Content: empty. No imports. Creates the `ddpg` package.

---

### `tests/test_actor.py`, `tests/test_critic.py`, `tests/test_agent.py`, `tests/test_replay_buffer.py` (test, —)

**Analog:** `tests/test_neural_renderer.py` (exact match)

**Imports pattern** (`tests/test_neural_renderer.py` lines 1–4):
```python
import torch
import pytest
from config import IMG_SIZE, STROKE_DIM
from models.renderer import SoftRasterizer, NeuralRenderer
```
Adapt per test file:
- `test_actor.py`: `from models.actor import Actor`
- `test_critic.py`: `from models.critic import Critic`
- `test_agent.py`: `from ddpg.agent import DDPGAgent, soft_update; from config import TAU`
- `test_replay_buffer.py`: `from ddpg.replay_buffer import ReplayBuffer; from config import REPLAY_BUFFER_CAPACITY, IMG_SIZE, STROKES_PER_STEP, STROKE_DIM`

**Shape assertion pattern** (`tests/test_neural_renderer.py` lines 7–10):
```python
def test_output_shape():
    R = SoftRasterizer()
    out = R(torch.rand(4, STROKE_DIM))
    assert out.shape == (4, 3, IMG_SIZE, IMG_SIZE)
```
Copy exactly: instantiate → forward with `torch.rand` → `assert out.shape == (...)`.

**Range assertion pattern** (`tests/test_neural_renderer.py` lines 13–16):
```python
def test_output_range():
    R = SoftRasterizer()
    out = R(torch.rand(32, STROKE_DIM))
    assert out.min() >= 0.0 and out.max() <= 1.0
```
Apply to actor (sigmoid output must be in [0,1]) and critic unbounded check (`not torch.isnan(out).any()`).

**eval + no_grad pattern** (`tests/test_neural_renderer.py` lines 19–24):
```python
def test_single_sample():
    R = SoftRasterizer()
    R.eval()
    with torch.no_grad():
        out = R(torch.rand(1, STROKE_DIM))
    assert out.shape == (1, 3, IMG_SIZE, IMG_SIZE)
```
Use `model.eval()` + `torch.no_grad()` in all shape tests. Tests should not require GPU.

**GPU skip pattern** (`tests/test_neural_renderer.py` lines 76–82):
```python
def test_gpu():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    device = torch.device("cuda")
    R = SoftRasterizer().to(device)
    out = R(torch.rand(1, STROKE_DIM, device=device))
    assert out.device.type == "cuda"
```
Copy this `pytest.skip("No CUDA")` pattern verbatim for all GPU tests.

**Module structure check pattern** (`tests/test_neural_renderer.py` lines 69–73):
```python
def test_no_batchnorm():
    R = SoftRasterizer()
    bn = [m for m in R.modules()
          if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
    assert len(bn) == 0
```
Invert for actor: assert BatchNorm2d IS present. For critic: assert BatchNorm2d is NOT present, and assert `weight_norm` parametrization IS present.

**Specific test cases for agent** (from RESEARCH.md code examples, soft update verification):
```python
def test_soft_update():
    from config import TAU
    agent = DDPGAgent(device=torch.device('cpu'))
    p_before = next(agent.critic_target.parameters()).data.clone()
    p_source  = next(agent.critic.parameters()).data
    soft_update(agent.critic_target, agent.critic, TAU)
    p_after   = next(agent.critic_target.parameters()).data
    expected  = (1 - TAU) * p_before + TAU * p_source
    assert torch.allclose(p_after, expected, atol=1e-6)

def test_target_eval_mode():
    agent = DDPGAgent(device=torch.device('cpu'))
    assert not agent.actor_target.training
    assert not agent.critic_target.training
```

---

## Shared Patterns

### Module init convention
**Source:** `models/renderer.py` lines 1–4 and `pretrain_renderer.py` lines 1–9
**Apply to:** All new `.py` files
```python
# Pattern: absolute imports from config, no relative imports
from config import IMG_SIZE, STROKE_DIM  # etc.
from models.renderer import SoftRasterizer  # when needed
```
Project uses flat absolute imports everywhere. No `from .config import` relative-style.

### register_buffer for device-agnostic tensors
**Source:** `models/renderer.py` lines 33–37
**Apply to:** `models/actor.py` and `models/critic.py` (CoordConv coordinate grids)
```python
self.register_buffer('xx', xx.contiguous())
self.register_buffer('yy', yy.contiguous())
```
All tensors that must follow `.to(device)` must be registered buffers, not plain attributes.

### Freeze pattern
**Source:** `pretrain_renderer.py` lines 63–69
**Apply to:** `ddpg/agent.py` target network init
```python
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
```
Both lines are mandatory: `eval()` disables BN/dropout behavior at test time; `requires_grad_(False)` prevents gradient accumulation.

### Config constants, no magic numbers
**Source:** `config.py` (all lines)
**Apply to:** All new files
```python
# All DDPG hyperparameters are in config.py:
# ACTOR_LR, CRITIC_LR, ACTOR_LR_FINAL, CRITIC_LR_FINAL, LR_DECAY_STEP
# GAMMA, TAU, BATCH_SIZE, REPLAY_BUFFER_CAPACITY, GRAD_CLIP_CRITIC
```
Never hardcode `0.005`, `200_000`, `0.955`, `96`, etc. Always import from `config`.

### Test file structure
**Source:** `tests/test_neural_renderer.py` (full file)
**Apply to:** All four new test files
- One test per requirement assertion
- Function names follow `test_<what_is_being_tested>()` pattern
- No fixtures — plain function-level instantiation
- `pytest.skip("No CUDA")` guard on all GPU tests
- No `if __name__ == "__main__"` block

---

## No Analog Found

| File | Role | Data Flow | Reason |
|---|---|---|---|
| `ddpg/replay_buffer.py` | utility | batch | No numpy ring buffer exists in the codebase yet |
| `ddpg/agent.py` (update_step) | service | request-response | No DDPG training loop exists yet; partial analog only in freeze helpers |

For these files, use RESEARCH.md Pattern 5 (ReplayBuffer) and Pattern 4 + Pattern 6 (Agent) directly.

---

## Critical Pitfalls (must note in plan actions)

1. **deepcopy + weight_norm:** Use `from torch.nn.utils.parametrizations import weight_norm` in `models/critic.py`. Never use `torch.nn.utils.weight_norm`. This is the only way to make `copy.deepcopy(critic)` safe in `ddpg/agent.py`.

2. **TReLU non-inplace:** `F.relu(x - self.alpha) + self.alpha` — never `inplace=True` here. Inplace on a tensor derived from a Parameter corrupts the autograd graph.

3. **Actor BN mode:** Actor must be in `train()` during gradient updates and `eval()` during rollout. Tests should call `.eval()` before shape checks to simulate rollout behavior.

4. **Step channel storage:** Store step as scalar float32 in replay buffer, tile to `(B, 1, H, W)` only in `sample()`. Never pre-tile and store as spatial channel (3.3 GB waste per obs/next_obs buffer).

5. **CoordConv input channels:** Actor/critic state is 7 channels. CoordConv appends 2 coord channels → Conv2d receives 9 channels. `CoordConv(in_channels=7, ...)` internally creates `Conv2d(9, 64, ...)`.

---

## Metadata

**Analog search scope:** `models/`, `tests/`, `pretrain_renderer.py`, `config.py`
**Files scanned:** 4 (renderer.py, test_neural_renderer.py, config.py, pretrain_renderer.py)
**Pattern extraction date:** 2026-06-10
