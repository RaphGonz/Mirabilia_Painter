# Phase 3: DDPG Models — Research

**Researched:** 2026-06-10
**Domain:** PyTorch DDPG — ResNet18 + CoordConv actor/critic, WN+TReLU critic, numpy replay buffer
**Confidence:** MEDIUM (architecture locked in CONTEXT.md; critical WN+deepcopy pitfall confirmed from PyTorch issues)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** State is 7 channels for BOTH actor and critic: `canvas(3) + target(3) + step_normalized(1)`.
- **D-02:** Step encoded as scalar `t/N_STROKES ∈ [0,1]`, broadcast to a full `(1,64,64)` spatial channel before concatenation.
- **D-03:** Critic receives `s_{t+1}` (rendered next-state), NOT `(s_t, a_t)`. Critic input shape: `(batch, 7, 64, 64)`.
- **D-04:** Actor first layer: 3×3 CoordConv stride-2 → `(batch, 64, 32, 32)`. CoordConv appends (x,y) coord channels before Conv2d, making input to Conv2d = 9 channels (7+2).
- **D-05:** Actor output: `FC(512, 40)` + `sigmoid` → `(batch, 40)` in `[0,1]`. `40 = k×STROKE_DIM = 5×8`.
- **D-06:** Actor uses BatchNorm. Runs in training mode during gradient updates (batch=96), eval() mode during rollout (batch=1 inference).
- **D-07:** Critic first layer: 3×3 CoordConv stride-2 → `(batch, 64, 32, 32)`.
- **D-08:** Critic uses WeightNorm + TReLU (NOT BatchNorm). WN via `torch.nn.utils.weight_norm()`.
- **D-09:** TReLU: learnable threshold. Reference impl formula: `F.relu(x - self.alpha) + self.alpha` where `self.alpha = nn.Parameter(torch.FloatTensor(1))` initialized to 0. One scalar alpha per TReLU instance (not per-channel).
- **D-10:** Target actor and critic = `copy.deepcopy()` at init, permanently in `eval()` mode.
- **D-11:** Soft update τ=0.005: `θ_target ← τ·θ + (1-τ)·θ_target`, after every critic gradient step.
- **D-12:** Target networks generate Bellman target: `y = r + γ·V_target(s_{t+1})`.
- **D-13:** Numpy ring buffer, pre-allocated. 5 conceptual fields: obs, act, rew, next_obs, done.
- **D-14:** Capacity: 200k transitions.
- **D-15:** Canvas channels stored as uint8 [0,255]; step scalar stored as float32 (not tiled). Sampling converts canvas uint8→float32 by dividing by 255.
- **D-16:** Target image stored redundantly per transition (no deduplication).
- **D-17:** `sample(batch_size)` returns float32 tensors with canvas channels normalized to `[0,1]`.
- **D-18:** Actor LR: 3e-4, decays to 1e-4 after 1e5 batches. Critic LR: 1e-3, decays to 3e-4.
- **D-19:** Discount γ=0.955 (γ^k, k=5). Phase 4 uses this directly.
- **D-20:** Batch size: 96.
- **D-21:** Critic gradient clipping max_norm=1.0. No clipping on actor.
- **D-22:** `ddpg/` directory for `agent.py` and `replay_buffer.py`. `models/actor.py` and `models/critic.py` alongside `models/renderer.py`.

### Claude's Discretion

None specified — all major decisions locked.

### Deferred Ideas (OUT OF SCOPE)

- WGAN discriminator (`models/discriminator.py`)
- LR scheduler implementation (Phase 4)
- CoordConv radius channel (paper doesn't use it)
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DDPG-01 | `models/actor.py`: input `(batch, 7, 64, 64)` → output `(batch, 40)` via sigmoid | ResNet18+CoordConv+BN architecture pattern verified from reference impl |
| DDPG-02 | `models/critic.py`: input `(batch, 7, 64, 64)` → scalar V(s') | ResNet18+CoordConv+WN+TReLU pattern; WN deepcopy pitfall documented |
| DDPG-03 | `ddpg/agent.py`: target networks (deepcopy), eval() permanent, soft update τ=0.005 | deepcopy+WN bug and workaround documented in Pitfall 1 |
| DDPG-04 | `ddpg/replay_buffer.py`: 200k capacity, uint8 canvas, float32 on sample | Memory layout verified: 9.87 GB total fits 31 GB RAM |
</phase_requirements>

---

## Summary

Phase 3 implements four self-contained modules: actor CNN, critic CNN, DDPG agent wrapper, and replay buffer. All architectural choices are locked in CONTEXT.md based on the "Learning to Paint" paper (ICCV 2019) — research here confirms implementation details, identifies one critical pitfall, and provides verified code patterns.

The actor and critic share a ResNet18-like backbone adapted for 64×64 input. Both start with a CoordConv layer (appending normalized x,y channels), followed by 4 residual stages [2,2,2,2] BasicBlocks. The actor uses standard BatchNorm; the critic replaces BatchNorm with WeightNorm+TReLU. Both end with global average pool + FC head.

The critical engineering pitfall is the `copy.deepcopy` + `torch.nn.utils.weight_norm` incompatibility: after init but before any forward pass, `weight.is_leaf=False` and deepcopy raises a RuntimeError. The mandatory workaround is to run a dummy forward pass under `torch.no_grad()` on the critic before calling `deepcopy`. Alternatively, use `torch.nn.utils.parametrizations.weight_norm` (modern API, deepcopy-safe, and also avoids the deprecation warning).

The replay buffer design stores 6-channel uint8 canvas arrays (4.92 GB each for obs and next_obs) plus float32 scalar step values — totaling ~9.87 GB pre-allocated on the machine's 31 GB RAM. This is well within budget.

**Primary recommendation:** Implement in this order: CoordConv helper → BasicBlock helper → Actor → Critic → Agent (with deepcopy fix) → ReplayBuffer. Test each with shape assertions before moving to the next.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Action generation (stroke params) | Actor CNN | — | Actor maps obs → action space, is the policy |
| State value estimation V(s') | Critic CNN | — | Critic evaluates rendered next-state quality |
| Target network management | Agent (ddpg/agent.py) | — | Agent owns the training loop scaffolding |
| Soft parameter update | Agent (ddpg/agent.py) | — | τ-weighted blend of current and target params |
| Experience storage | Replay Buffer | — | Independent, no model dependency |
| Differentiable rendering (for actor loss) | SoftRasterizer (frozen) | — | Stays in Phase 4; Phase 3 only scaffolds the call |

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.7.0 | ResNet, CoordConv, WN, BN, Adam, soft update | Only PyTorch — project constraint |
| torch.nn.utils.parametrizations | 2.7.0 (built-in) | weight_norm for critic Conv2d layers | Modern API; deepcopy-safe; avoids deprecated `torch.nn.utils.weight_norm` |
| numpy | 1.26.x or 2.0.x | Pre-allocated ring buffer arrays (uint8, float32) | Required for O(1) buffer ops without Python list overhead |

### Supporting

No new packages are required for this phase. All functionality comes from PyTorch builtins and numpy.

**Installation:** No new packages. Existing environment is sufficient.

---

## Package Legitimacy Audit

No new packages are introduced in Phase 3. All components use `torch` (already installed), `numpy` (already installed), and Python standard library (`copy`, `collections`). No audit required.

---

## Architecture Patterns

### System Architecture Diagram

```
obs (7×64×64)
     │
     ▼
[CoordConv stride-2]   →  append (x,y) channels  →  9ch input to Conv2d(9,64,k=3,s=2)
     │                    (registered buffers)
     ▼
(64 × 32 × 32)
     │
     ▼
[ResNet18 Backbone]
  Stage 1: 2×BasicBlock(64→64)         →  (64 × 32 × 32)
  Stage 2: 2×BasicBlock(64→128, s=2)   →  (128 × 16 × 16)
  Stage 3: 2×BasicBlock(128→256, s=2)  →  (256 × 8 × 8)
  Stage 4: 2×BasicBlock(256→512, s=2)  →  (512 × 4 × 4)
     │
     ▼
[Global Average Pool]  →  (512,)
     │
     ▼
[FC(512, 40)] + sigmoid    ← Actor
[FC(512, 1)]               ← Critic (no activation)
```

Actor uses BatchNorm2d after each Conv2d in backbone.
Critic uses weight_norm on each Conv2d + TReLU activation instead of BN+ReLU.

### Recommended Project Structure

```
models/
├── __init__.py          # already exists (empty)
├── renderer.py          # already exists (SoftRasterizer)
├── actor.py             # NEW: ResNet18+CoordConv+BN → (batch, 40) sigmoid
└── critic.py            # NEW: ResNet18+CoordConv+WN+TReLU → (batch, 1)
ddpg/
├── __init__.py          # NEW: empty, makes ddpg a package
├── agent.py             # NEW: target networks, soft update, update_step scaffold
└── replay_buffer.py     # NEW: numpy ring buffer, uint8 storage
tests/
├── test_actor.py        # NEW: shape assertions for DDPG-01
├── test_critic.py       # NEW: shape assertions for DDPG-02
├── test_agent.py        # NEW: deepcopy, soft update, eval mode for DDPG-03
└── test_replay_buffer.py # NEW: capacity, uint8 storage, sampling shapes for DDPG-04
```

### Pattern 1: CoordConv Layer

**What:** Prepend normalized (x,y) coordinate channels to the spatial input before Conv2d.
**When to use:** First layer of both actor and critic, per CONTEXT.md D-04/D-07.

```python
# Source: mkocabas/CoordConv-pytorch + adaptation for register_buffer device tracking
import torch
import torch.nn as nn
from config import IMG_SIZE

class CoordConv(nn.Module):
    """
    Wraps a Conv2d to prepend normalized (x,y) coordinate channels.
    Input shape:  (B, in_channels, H, W)
    Output shape: (B, out_channels, H', W')  [H', W' depend on stride/padding]
    The Conv2d inside receives (in_channels + 2) input channels.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 bias: bool = False):
        super().__init__()
        # Build normalized coordinate grids once; register as buffers so .to(device) works
        H = W = IMG_SIZE  # grids are for the input spatial size
        # xx: x-coord normalized to [-1, 1], shape (1, 1, H, W)
        xx = torch.linspace(-1, 1, W).view(1, 1, 1, W).expand(1, 1, H, W)
        # yy: y-coord normalized to [-1, 1], shape (1, 1, H, W)
        yy = torch.linspace(-1, 1, H).view(1, 1, H, 1).expand(1, 1, H, W)
        self.register_buffer('xx', xx.contiguous())
        self.register_buffer('yy', yy.contiguous())
        # The actual Conv2d takes in_channels + 2 input channels
        self.conv = nn.Conv2d(in_channels + 2, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        xx = self.xx.expand(B, -1, -1, -1)  # (B, 1, H, W)
        yy = self.yy.expand(B, -1, -1, -1)  # (B, 1, H, W)
        x = torch.cat([x, xx, yy], dim=1)   # (B, in_channels+2, H, W)
        return self.conv(x)                  # (B, out_channels, H', W')
```

**Key point:** `in_channels=7` → Conv2d receives 9 channels. Buffers ensure coordinates move with `.to(device)`.

### Pattern 2: ResNet18 BasicBlock (Actor variant — with BatchNorm)

```python
# Source: Adapted from torchvision ResNet (github.com/pytorch/vision) for custom in_channels
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample=None, use_bn: bool = True):
        super().__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.downsample = downsample  # 1×1 conv for shortcut when dims change

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)
```

**Downsample construction** (when stride>1 or channel change):
```python
downsample = nn.Sequential(
    nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
    nn.BatchNorm2d(out_channels),  # or nn.Identity() for critic
)
```

### Pattern 3: Critic BasicBlock (WeightNorm + TReLU variant)

```python
# Source: Reference impl hzwer/ICCV2019-LearningToPaint/baseline/DRL/critic.py
from torch.nn.utils.parametrizations import weight_norm  # modern API — deepcopy-safe

class TReLU(nn.Module):
    """Translated ReLU: F.relu(x - alpha) + alpha, scalar alpha per instance."""
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return F.relu(x - self.alpha) + self.alpha

class BasicBlockWN(nn.Module):
    """BasicBlock with WeightNorm+TReLU instead of BatchNorm+ReLU (for critic)."""
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample=None):
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

**Note:** `torch.nn.utils.parametrizations.weight_norm` (not the deprecated `torch.nn.utils.weight_norm`) is used here specifically to avoid the deepcopy bug documented in Pitfall 1.

### Pattern 4: Soft Update

```python
# Source: standard DDPG pattern (OpenAI, SB3, reference impl)
@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for p_targ, p in zip(target.parameters(), source.parameters()):
        p_targ.data.mul_(1.0 - tau)
        p_targ.data.add_(tau * p.data)
```

**Call sequence per training step:**
1. Update critic (backward + step)
2. Update actor (backward + step)
3. `soft_update(actor_target, actor, TAU)`
4. `soft_update(critic_target, critic, TAU)`

### Pattern 5: Replay Buffer numpy ring buffer

```python
# [ASSUMED] — standard pattern from SB3 / community DDPG implementations
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int, img_size: int = 64):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        # 6ch canvas (3ch canvas + 3ch target) stored uint8
        self.obs_canvas    = np.zeros((capacity, 6, img_size, img_size), dtype=np.uint8)
        self.next_canvas   = np.zeros((capacity, 6, img_size, img_size), dtype=np.uint8)
        # step scalar stored as float32 (NOT tiled 64×64 — 4096× memory savings)
        self.obs_step      = np.zeros((capacity,), dtype=np.float32)
        self.next_step     = np.zeros((capacity,), dtype=np.float32)
        # action, reward, done
        self.actions       = np.zeros((capacity, 40), dtype=np.float32)
        self.rewards       = np.zeros((capacity,), dtype=np.float32)
        self.dones         = np.zeros((capacity,), dtype=bool)

    def push(self, obs_canvas, obs_step, act, rew, next_canvas, next_step, done):
        idx = self.ptr
        self.obs_canvas[idx]  = obs_canvas   # uint8 (6, H, W)
        self.obs_step[idx]    = obs_step      # float32 scalar
        self.actions[idx]     = act           # float32 (40,)
        self.rewards[idx]     = rew
        self.next_canvas[idx] = next_canvas   # uint8 (6, H, W)
        self.next_step[idx]   = next_step     # float32 scalar
        self.dones[idx]       = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        # Reconstruct 7ch obs: normalize 6ch uint8 → float32, then append step channel
        canvas = torch.from_numpy(self.obs_canvas[idx]).float().div(255.0).to(device)
        step   = torch.from_numpy(self.obs_step[idx]).to(device)
        # tile step scalar → (B, 1, H, W) spatial channel
        H, W = canvas.shape[-2], canvas.shape[-1]
        step_ch  = step.view(-1, 1, 1, 1).expand(-1, 1, H, W)
        obs = torch.cat([canvas, step_ch], dim=1)   # (B, 7, 64, 64)

        n_canvas = torch.from_numpy(self.next_canvas[idx]).float().div(255.0).to(device)
        n_step   = torch.from_numpy(self.next_step[idx]).to(device)
        n_step_ch = n_step.view(-1, 1, 1, 1).expand(-1, 1, H, W)
        next_obs = torch.cat([n_canvas, n_step_ch], dim=1)   # (B, 7, 64, 64)

        act  = torch.from_numpy(self.actions[idx]).to(device)
        rew  = torch.from_numpy(self.rewards[idx]).to(device)
        done = torch.from_numpy(self.dones[idx]).to(device)
        return obs, act, rew, next_obs, done
```

**Memory:** 200k × 6 × 64 × 64 × 1 (uint8) × 2 (obs + next) = 9.84 GB. Machine has 31 GB RAM — fits comfortably. Step scalars add negligible 1.6 MB.

### Pattern 6: Target Network Init (deepcopy-safe)

```python
import copy

# After building critic WITH parametrizations.weight_norm (deepcopy-safe):
critic = Critic(...)
# DO NOT use: torch.nn.utils.weight_norm (deprecated, deepcopy bug)
# DO use: torch.nn.utils.parametrizations.weight_norm (applied inside Critic.__init__)

critic_target = copy.deepcopy(critic)   # Safe with parametrizations API
critic_target.eval()
for p in critic_target.parameters():
    p.requires_grad_(False)

actor_target = copy.deepcopy(actor)
actor_target.eval()
for p in actor_target.parameters():
    p.requires_grad_(False)
```

### Anti-Patterns to Avoid

- **Using `torch.nn.utils.weight_norm` (deprecated):** Causes deepcopy to fail. Use `torch.nn.utils.parametrizations.weight_norm` instead.
- **Calling `actor.eval()` for the actor update pass:** Actor must be in `train()` mode during the policy gradient update so BatchNorm updates its running stats. Call `actor.eval()` only for rollout/inference.
- **Tiling the step channel in the replay buffer:** Storing `(capacity, 1, 64, 64)` float32 costs 3.3 GB extra. Store as scalar, tile during `sample()`.
- **Using `in_place=True` on TReLU output:** `F.relu(..., inplace=True)` on the result of `self.alpha` subtraction can corrupt the graph. Use non-inplace version.
- **Adding a `bias=True` to weight-normed Conv2d:** The WN reparameterization replaces the weight tensor; bias is a separate parameter and is fine, but the reference implementation uses `bias=False` throughout.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Weight reparameterization | Custom weight scaling | `torch.nn.utils.parametrizations.weight_norm` | Handles hook lifecycle, state_dict compat, deepcopy correctly |
| Soft parameter update | Loop with manual lerp | `p_targ.data.mul_(1-tau).add_(tau*p.data)` | In-place ops avoid unnecessary tensor allocation |
| Coordinate channels | Any custom spatial encoding | CoordConv pattern (Pattern 1 above) | Standard, tested, device-agnostic via register_buffer |
| Ring buffer indexing | Python list + modulo | numpy pre-allocated + `ptr % capacity` | Python list grows dynamically; numpy is O(1) insert and sample |

**Key insight:** The entire phase uses PyTorch builtins and numpy. No third-party RL libraries are needed or permitted (per CLAUDE.md — no SB3, TorchRL, RLlib).

---

## Common Pitfalls

### Pitfall 1: deepcopy + weight_norm crash (CRITICAL)

**What goes wrong:** `copy.deepcopy(critic)` raises `RuntimeError: Only Tensors created explicitly by the user (graph leaves) support deepcopy at the moment` immediately after critic is initialized with `torch.nn.utils.weight_norm`.

**Why it happens:** After `weight_norm(conv)`, `conv.weight` is a computed tensor (output of the forward pre-hook), not a leaf. `is_leaf=False`. PyTorch's deepcopy protocol requires leaf tensors.

**How to avoid:** Use `torch.nn.utils.parametrizations.weight_norm` (modern API, `from torch.nn.utils.parametrizations import weight_norm`). This is deepcopy-safe. Do NOT use the deprecated `torch.nn.utils.weight_norm`.

**Fallback if old API required:** Run `with torch.no_grad(): _ = critic(dummy_input)` before deepcopy — this materializes the hook output as a leaf tensor, setting `weight.is_leaf=True`.

**Warning signs:** Crash on the `copy.deepcopy(critic)` line in `agent.py __init__`.

**References:** [GitHub issue #28594](https://github.com/pytorch/pytorch/issues/28594), [issue #102981](https://github.com/pytorch/pytorch/issues/102981)

### Pitfall 2: Actor BatchNorm at batch_size=1 during rollout

**What goes wrong:** If actor remains in `train()` mode during env.step() rollout, BatchNorm2d computes mean/variance over a batch of 1, producing NaN variance → NaN output → NaN action.

**Why it happens:** BN variance of a single sample is undefined (denominator N-1=0 with standard variance).

**How to avoid:** Always call `actor.eval()` before rollout and `actor.train()` before the gradient update step. Running stats accumulated during training are used during eval — safe for batch=1.

**Warning signs:** NaN actions or all-zero actions during env rollout early in training.

### Pitfall 3: Target network gradient accumulation

**What goes wrong:** Gradients flow into target network parameters, contaminating the training signal and causing instability.

**Why it happens:** Forgetting `requires_grad_(False)` on target network parameters, or calling `backward()` while target network is on the compute graph.

**How to avoid:** After deepcopy, immediately call `for p in target.parameters(): p.requires_grad_(False)`. Additionally use `torch.no_grad()` context when computing Bellman targets.

**Warning signs:** Target network parameters change unexpectedly between updates (can test by printing param norm before/after update).

### Pitfall 4: Reward target image stored per transition (D-16) — memory cost

**What goes wrong:** Storing 3-channel target image per transition in the buffer doubles canvas storage. 200k × 6 × 64 × 64 uint8 = 9.84 GB. If naively stored as float32, it would be 39 GB.

**Why it happens:** D-16 says store redundantly for simplicity.

**How to avoid:** Store canvas (3ch) + target (3ch) together as 6ch uint8 — this IS the D-15 design. Never expand to float32 in the buffer. Only convert on `sample()`.

**Warning signs:** `MemoryError` or `numpy.core._exceptions._ArrayMemoryError` at buffer initialization.

### Pitfall 5: TReLU scalar alpha vs per-channel

**What goes wrong:** The CONTEXT.md D-09 says "per-channel bias". The actual reference implementation uses a **scalar** `nn.Parameter(torch.FloatTensor(1))` — one alpha per TReLU instance (not per channel).

**Why it happens:** Description mismatch between the context document and the reference code.

**How to avoid:** Use scalar alpha as in the reference (`torch.zeros(1)`). One TReLU instance per activation position in the block. This produces one learnable threshold per "slot" in the network (after conv1, after conv2, after shortcut addition — per BasicBlockWN).

**Warning signs:** Shape mismatch if you try to multiply per-channel alpha against a (B,C,H,W) tensor without proper broadcasting.

### Pitfall 6: ResNet18 stage channel count for 64×64 input

**What goes wrong:** Standard ResNet18 uses a 7×7 conv with stride 2 + maxpool stride 2 as the stem, reducing 224×224 → 56×56. For 64×64, this is too aggressive.

**Why it happens:** Blindly copying torchvision's resnet18.

**How to avoid:** The CONTEXT design already avoids this: CoordConv stride-2 reduces 64→32; the 4 ResNet stages then reduce 32→16→8→4→global average pool. No maxpool. This matches the architecture described in the paper appendix for the smaller input resolution.

**Warning signs:** Spatial dimensions going to 1×1 before global average pool, or strided convolutions producing 0-size feature maps.

---

## Code Examples

### Actor: full forward pass shape trace

```python
# Source: derived from CONTEXT.md D-04/D-05 + Pattern 1/2 above
# Input: (batch, 7, 64, 64)
# CoordConv: 7ch + 2 coord = 9ch input → Conv2d(9, 64, 3, stride=2, pad=1) → (batch, 64, 32, 32)
# Stage 1: BasicBlock(64,64)×2       → (batch, 64, 32, 32)  [no stride change]
# Stage 2: BasicBlock(64,128,s=2)×2  → (batch, 128, 16, 16)
# Stage 3: BasicBlock(128,256,s=2)×2 → (batch, 256, 8, 8)
# Stage 4: BasicBlock(256,512,s=2)×2 → (batch, 512, 4, 4)
# AdaptiveAvgPool2d(1, 1)            → (batch, 512, 1, 1)
# Flatten                            → (batch, 512)
# Linear(512, 40) + sigmoid          → (batch, 40)

# Quick shape assertion test:
import torch
actor = Actor()  # ResNet18+CoordConv+BN
x = torch.zeros(2, 7, 64, 64)
out = actor(x)
assert out.shape == (2, 40), f"Expected (2,40), got {out.shape}"
assert out.min() >= 0.0 and out.max() <= 1.0
```

### Critic: full forward pass shape trace

```python
# Input: (batch, 7, 64, 64)  -- s_{t+1}, same structure as actor input
# CoordConv: 7+2=9ch → Conv2d(9, 64, 3, stride=2, pad=1) → (batch, 64, 32, 32)
# Stages 1-4 (same reduction as actor, but WN+TReLU instead of BN+ReLU)
# AdaptiveAvgPool2d(1, 1) → (batch, 512, 1, 1)
# Flatten → (batch, 512)
# Linear(512, 1) [no activation] → (batch, 1)

critic = Critic()
x = torch.zeros(2, 7, 64, 64)
out = critic(x)
assert out.shape == (2, 1), f"Expected (2,1), got {out.shape}"
assert not torch.isnan(out).any()
```

### Agent: soft update verification

```python
from config import TAU
# After one soft_update call, target params should be:
# p_targ = (1-TAU)*p_targ_old + TAU*p
# With TAU=0.005, after 1 update from identical init, no change
# After 1 update where source differs by delta, target moves by TAU*delta
p_before = next(critic_target.parameters()).data.clone()
p_source  = next(critic.parameters()).data
soft_update(critic_target, critic, TAU)
p_after = next(critic_target.parameters()).data
expected = (1 - TAU) * p_before + TAU * p_source
assert torch.allclose(p_after, expected, atol=1e-6)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.nn.utils.weight_norm` | `torch.nn.utils.parametrizations.weight_norm` | PyTorch 1.9+ | deepcopy-safe; modern API; compat with state_dict |
| OU noise for DDPG exploration | Gaussian noise (σ annealed) | ~2019-2021 | Simpler; adequate for painting task (no temporal correlation needed) |
| 7×7 stem + MaxPool for ResNet | CoordConv stride-2 only (no MaxPool) | Paper design for small inputs | Preserves spatial resolution at 64×64 |

**Deprecated/outdated:**
- `torch.nn.utils.weight_norm`: deprecated since PyTorch 1.9; use `parametrizations.weight_norm`.
- MLP-based renderer (NeuralRenderer): replaced by SoftRasterizer (analytical, no pretraining) in Phase 2.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | TReLU alpha is a scalar (torch.zeros(1)), not per-channel | Pitfall 5, Pattern 3 | Shape error if per-channel broadcasting assumed; LOW risk — reference code is explicit |
| A2 | `torch.nn.utils.parametrizations.weight_norm` is deepcopy-safe in PyTorch 2.7 | Pitfall 1, Pattern 3 | If not fixed, need Workaround A (dummy forward before deepcopy) |
| A3 | Replay buffer memory of ~9.87 GB fits within available RAM without OOM during training | Pattern 5 | If model weights + gradients + replay buffer exceed 31 GB, reduce REPLAY_BUFFER_CAPACITY |
| A4 | BasicBlock downsample for critic uses `weight_norm(Conv2d(...))` + TReLU | Pattern 2/3 | If WN not applied to downsample convs, critic training may be less stable — LOW severity |

---

## Open Questions

1. **ddpg/agent.py scope in Phase 3**
   - What we know: CONTEXT.md D-22 says agent.py is in Phase 3 scope for target network management and soft update.
   - What's unclear: Phase 3 vs Phase 4 boundary for `update_step()` — actor loss requires SoftRasterizer (frozen) which is Phase 4. CONTEXT.md Phase 3 boundary says "actor/critic update step" is Phase 3.
   - Recommendation: Implement `update_step()` scaffold in Phase 3 (structure + shapes + soft update), but leave SoftRasterizer integration as a TODO comment. Phase 4 fills it in during env.py/train.py work.

2. **Actor BN mode during parallel env rollout (Phase 4 concern)**
   - What we know: 96 parallel envs batch obs together → actor sees batch of 96 during collection → BN is fine.
   - What's unclear: Whether actor will ever be called with batch=1 (single env eval).
   - Recommendation: Always call `actor.eval()` during eval rollout; `actor.train()` during policy gradient updates. This is standard DDPG practice.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | All modules | Yes | 3.11 (assumed) | — |
| torch | Actor, critic, agent | Yes (Phase 2 used it) | 2.7.0 | — |
| numpy | Replay buffer | Yes | 1.26+ or 2.0+ | — |
| pytest | Tests | Yes (pyproject.toml present) | 7.x+ | — |
| CUDA | GPU training | Yes (Phase 2 confirmed) | 12.x | CPU fallback for tests |

No missing dependencies.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `pyproject.toml` (`testpaths = ["tests"]`) |
| Quick run command | `pytest tests/test_actor.py tests/test_critic.py tests/test_agent.py tests/test_replay_buffer.py -x -q` |
| Full suite command | `pytest tests/ -x -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DDPG-01 | Actor: (B,7,64,64) → (B,40) in [0,1] | unit | `pytest tests/test_actor.py -x -q` | Wave 0 |
| DDPG-01 | Actor: no output outside [0,1] range | unit | `pytest tests/test_actor.py::test_output_range -x` | Wave 0 |
| DDPG-02 | Critic: (B,7,64,64) → (B,1) scalar | unit | `pytest tests/test_critic.py::test_critic_shape -x` | Wave 0 |
| DDPG-02 | Critic: no sigmoid on output (unbounded) | unit | `pytest tests/test_critic.py::test_critic_unbounded -x` | Wave 0 |
| DDPG-03 | Target networks in eval() permanently | unit | `pytest tests/test_agent.py::test_target_eval_mode -x` | Wave 0 |
| DDPG-03 | Soft update: target params are weighted avg | unit | `pytest tests/test_agent.py::test_soft_update -x` | Wave 0 |
| DDPG-04 | Buffer stores 200k transitions, uint8 canvas | unit | `pytest tests/test_replay_buffer.py::test_capacity -x` | Wave 0 |
| DDPG-04 | Buffer sample returns float32 tensors, correct shapes | unit | `pytest tests/test_replay_buffer.py::test_sample_shapes -x` | Wave 0 |

### Sampling Rate

- **Per task commit:** `pytest tests/test_actor.py tests/test_critic.py tests/test_agent.py tests/test_replay_buffer.py -x -q`
- **Per wave merge:** `pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/test_actor.py` — covers DDPG-01 shape and range assertions
- [ ] `tests/test_critic.py` — covers DDPG-02 shape and unbounded output
- [ ] `tests/test_agent.py` — covers DDPG-03 target eval mode, soft update correctness
- [ ] `tests/test_replay_buffer.py` — covers DDPG-04 capacity, uint8 storage, sample shapes
- [ ] `ddpg/__init__.py` — empty init file to make ddpg a package

---

## Security Domain

> `security_enforcement: true` in config. ASVS Level 1.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Local ML training, no auth surface |
| V3 Session Management | No | No sessions — single-process training |
| V4 Access Control | No | No multi-user, no access control surface |
| V5 Input Validation | Low | Replay buffer: numpy array bounds enforced by pre-allocation. Actions clamped to [0,1] by sigmoid. |
| V6 Cryptography | No | No secrets, no encryption needed |

### Known Threat Patterns for PyTorch ML Training

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| NaN gradient explosion | Tampering (data integrity) | Gradient clipping `max_norm=1.0` on critic (D-21) |
| Buffer overread | Tampering | numpy pre-allocation with fixed dtype — no dynamic resize |
| Model weight file tampering | Elevation | `.pt` checkpoints loaded only from local filesystem (Phase 4) |

No significant security surface for this phase — it is a pure local training component with no network access, no user input, and no authentication.

---

## Sources

### Primary (MEDIUM confidence — websearch verified against official sources)
- PyTorch issues #28594 and #102981 — `weight_norm` deepcopy bug, confirmed reproducible
- `https://github.com/hzwer/ICCV2019-LearningToPaint/blob/master/baseline/DRL/critic.py` — TReLU formula, WN pattern
- `https://github.com/hzwer/ICCV2019-LearningToPaint/blob/master/baseline/DRL/actor.py` — ResNet18 structure, BN use
- `https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py` — CoordConv coordinate normalization pattern

### Secondary (LOW confidence — websearch)
- PyTorch docs `torch.nn.utils.parametrizations.weight_norm` deprecation and compatibility notice
- PyTorch forums: BatchNorm2d eval mode at batch_size=1 behavior (running_mean/var used)

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — pure PyTorch builtins and numpy, already in environment
- Architecture: HIGH — locked decisions from CONTEXT.md + verified against reference implementation
- WN deepcopy pitfall: MEDIUM — multiple PyTorch issues confirm the bug; workaround confirmed; PyTorch 2.7 fix status not verified but `parametrizations.weight_norm` is the documented modern replacement
- TReLU formula: MEDIUM — verified against reference code (scalar alpha, not per-channel)
- Replay buffer memory: HIGH — computed from actual numpy allocation test on this machine (31 GB RAM)

**Research date:** 2026-06-10
**Valid until:** 2026-09-10 (stable — PyTorch builtins, no fast-moving dependencies)
