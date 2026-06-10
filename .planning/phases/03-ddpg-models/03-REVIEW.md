---
phase: 03-ddpg-models
reviewed: 2026-06-10T00:00:00Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - ddpg/agent.py
  - ddpg/replay_buffer.py
  - models/actor.py
  - models/critic.py
  - tests/test_actor.py
  - tests/test_agent.py
  - tests/test_critic.py
  - tests/test_replay_buffer.py
findings:
  critical: 3
  warning: 6
  info: 3
  total: 12
status: issues_found
---

# Phase 03: Code Review Report

**Reviewed:** 2026-06-10T00:00:00Z
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Reviewed the DDPG scaffolding for phase 3: actor/critic model definitions, replay buffer, agent lifecycle, and corresponding tests. The architecture choices (CoordConv stem, ResNet18-like backbone, WN+TReLU for critic, BN for actor, memory-efficient uint8 buffer) are sound and well-documented. Three critical defects were found: the Phase 4 update pseudocode has the live critic predicting the wrong state (the Bellman regression will be broken), the soft-update is silently a no-op on the critic target due to weight-norm parametrization internals, and `expand()` without `.contiguous()` before `torch.cat` is a correctness risk. Six warnings cover the lack of sampling guard in the buffer, a bool/float dtype mismatch on `done`, the hardcoded CoordConv grid size, BatchNorm single-sample inference issues in the actor, a tautological soft-update test, and a weak range assertion in the buffer test. Three info items cover a misleading docstring, an unused import in `nn`, and an unused `import copy` in the test.

---

## Critical Issues

### CR-01: Live critic receives `next_obs` instead of `obs` — Bellman regression broken

**File:** `ddpg/agent.py:92`
**Issue:** The Phase 4 pseudocode computes the bootstrap target `y` from `critic_target(next_obs)` (correct), then computes the live prediction with `v_pred = critic(next_obs)` — the same `next_obs`. A Bellman update for a state-value function must regress `V(s)` toward `r + γ * V(s')`. Using `next_obs` for both sides means the live critic learns to predict its own target, which is a degenerate regression that will not converge to the true value function. The actor loss downstream will receive a wrong Q-signal.

**Fix:**
```python
# Correct Bellman target computation
with torch.no_grad():
    v_next = critic_target(next_obs)       # V(s') from frozen target
    y = rew.unsqueeze(1) + GAMMA * v_next * (~done.unsqueeze(1).float())
v_pred = critic(obs)                       # V(s) from live critic — NOT next_obs
critic_loss = F.mse_loss(v_pred, y)
```

---

### CR-02: `soft_update` is silently a no-op on the critic target due to weight-norm parametrization

**File:** `ddpg/agent.py:22-24`
**Issue:** After `copy.deepcopy(critic)`, the `critic_target` retains the full weight-norm parametrization: each Conv2d stores `weight_g` and `weight_v` tensors, and the derived `weight` buffer is *recomputed* from them on every forward pass via the parametrization hook. `soft_update` mutates `p_targ.data` directly on the *derived* `weight` parameter. This write is overwritten by the next forward call, which recomputes `weight` from unchanged `weight_g`/`weight_v`. The actual persistent parameters that need updating are `weight_g` and `weight_v` — not `weight`. As a result, the critic target never actually soft-updates, making it identical to its initial state for the entire training run.

**Fix:**
```python
@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    # named_parameters() on a parametrized module yields the underlying
    # weight_g, weight_v etc. — NOT the derived 'weight'.
    # Use state_dict / load_state_dict for a safe, parametrization-aware update.
    target_sd = target.state_dict()
    source_sd = source.state_dict()
    for key in target_sd:
        target_sd[key].mul_(1.0 - tau).add_(tau * source_sd[key])
    target.load_state_dict(target_sd)
```
Alternatively, iterate `target.named_parameters()` and `source.named_parameters()` which yields `weight_g` and `weight_v` as distinct named entries; or bypass parametrizations by iterating `target._parameters` directly. The `state_dict` approach is safest and most future-proof.

---

### CR-03: `expand()` without `.contiguous()` before `torch.cat` — potential silent corruption

**File:** `ddpg/replay_buffer.py:112,118`
**Issue:** `step.view(-1, 1, 1, 1).expand(-1, 1, H, W)` returns a tensor with a zero stride in the H and W dimensions. When this non-contiguous, stride-0 tensor is passed to `torch.cat`, PyTorch internally calls `.contiguous()` on each input before concatenating. In practice this is usually correct, but on certain CUDA kernel paths (particularly with `torch.compile` or custom CUDA extensions downstream) a stride-0 tensor entering a concat can expose incorrect assumptions about memory layout. The risk is real if this code is ever wrapped with `torch.compile` for training speed.

**Fix:**
```python
# Current obs
step_ch = step.view(-1, 1, 1, 1).expand(-1, 1, H, W).contiguous()
obs = torch.cat([canvas, step_ch], dim=1)   # (B, 7, H, W)

# Next obs
n_step_ch = n_step.view(-1, 1, 1, 1).expand(-1, 1, H, W).contiguous()
next_obs = torch.cat([n_canvas, n_step_ch], dim=1)
```

---

## Warnings

### WR-01: `sample()` raises cryptic `ValueError` when buffer is empty, silently biases batches when `size < batch_size`

**File:** `ddpg/replay_buffer.py:104`
**Issue:** `np.random.randint(0, self.size, size=batch_size)` raises `ValueError: low >= high` when `self.size == 0`. When `0 < self.size < batch_size`, sampling with replacement silently returns repeated indices, producing a biased batch. There is no guard. Callers must track this externally or risk a confusing crash or biased gradient.

**Fix:**
```python
def sample(self, batch_size: int, device: torch.device):
    if self.size < batch_size:
        raise ValueError(
            f"Cannot sample {batch_size} transitions from buffer with only {self.size} stored."
        )
    idx = np.random.randint(0, self.size, size=batch_size)
    ...
```

---

### WR-02: `done` tensor is `torch.bool`, not `torch.float32` — arithmetic in Bellman update is fragile

**File:** `ddpg/replay_buffer.py:124`
**Issue:** `self.dones` is `dtype=bool`, so `torch.from_numpy(self.dones[idx])` yields a `torch.bool` tensor. The Bellman pseudocode in `agent.py:90` writes `GAMMA * v_next * (~done.unsqueeze(1))`. Multiplying a float tensor by a bool tensor triggers an implicit cast in recent PyTorch versions, but this behaviour is not guaranteed across versions and the code raises a `RuntimeError` in strict type-checking environments. The docstring says `done: (B,) bool` — but all other returned tensors are `float32`. The done flag should be returned as `float32` (0.0 / 1.0) for consistent arithmetic.

**Fix:**
```python
# In sample():
done = torch.from_numpy(self.dones[idx]).float().to(device)   # bool -> float32

# Storage dtype can stay bool (saves memory); only the returned tensor changes.
```

---

### WR-03: CoordConv grids are fixed to `IMG_SIZE` at construction — hardcoded spatial assumption

**File:** `models/actor.py:22-27`
**Issue:** `H = W = IMG_SIZE` is computed once at `__init__` and baked into registered buffers. If the model is ever called with an input of different spatial size (resolution ablation, higher-res episode), the shapes will mismatch inside `forward()` at `torch.cat([x, xx, yy], dim=1)`. The buffer shapes will be `(1, 1, 64, 64)` while `x` may be `(B, C, 128, 128)`, causing a shape error. This also affects `models/critic.py` since it imports and reuses `CoordConv`.

**Fix:**
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, _, H, W = x.shape
    xx = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
    yy = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
    x = torch.cat([x, xx, yy], dim=1)
    return self.conv(x)
    # Remove registered buffers from __init__ if switching to dynamic grids.
```

---

### WR-04: Actor backbone uses BatchNorm — single-sample inference during rollout uses unreliable running stats

**File:** `models/actor.py:55-56` (within `BasicBlock`)
**Issue:** CLAUDE.md explicitly warns "No batch norm... BN interacts poorly with single-sample inference at test time... Use layer norm or group norm if normalization is needed." The actor runs with `batch_size=1` during environment rollout (`actor.eval()` + single obs). In eval mode, BatchNorm uses accumulated running mean/var. Early in training these stats are unreliable, causing incorrect normalization for single-step inference. The renderer was designed without BN for this exact reason; the actor has the same vulnerability. The `test_actor_has_batchnorm` test actively asserts BN presence, which locks in this design choice.

**Fix:** Replace `BasicBlock` (BN) with `BasicBlockWN` (weight norm + TReLU) in the actor, or use `GroupNorm(num_groups=8, num_channels=out_ch)` as a drop-in BN replacement that is batch-size-independent. Update `test_actor_has_batchnorm` to become `test_actor_no_batchnorm` to match the convention used in `test_critic.py`.

---

### WR-05: `test_soft_update` is a tautology — source and target params are equal at init

**File:** `tests/test_agent.py:39-53`
**Issue:** At `DDPGAgent.__init__`, `critic_target` is a `deepcopy` of `critic`, so their parameters are bitwise equal. The test reads `p_before = next(agent.critic_target.parameters()).data.clone()` and `p_source = next(agent.critic.parameters()).data`. Since `p_before == p_source`, the expected value `(1-TAU)*p_before + TAU*p_source == p_before`. The assertion `torch.allclose(p_after, expected)` passes trivially even if `soft_update` is a no-op, because the expected value is unchanged. This test does not detect the CR-02 bug above.

**Fix:**
```python
def test_soft_update():
    agent = DDPGAgent(device=torch.device('cpu'))
    # Perturb live critic params to break the init-equal symmetry
    with torch.no_grad():
        for p in agent.critic.parameters():
            p.add_(torch.ones_like(p))   # shift all params by +1
    p_before = next(agent.critic_target.parameters()).data.clone()
    p_source = next(agent.critic.parameters()).data.clone()
    soft_update(agent.critic_target, agent.critic, TAU)
    p_after = next(agent.critic_target.parameters()).data
    expected = (1 - TAU) * p_before + TAU * p_source
    assert torch.allclose(p_after, expected, atol=1e-6)
```

---

### WR-06: `test_sample_dtype_and_range` validates only aggregate bounds — step channel goes unchecked

**File:** `tests/test_replay_buffer.py:140-156`
**Issue:** `obs.max() <= 1.0` and `obs.min() >= 0.0` are asserted on the full 7-channel tensor. The test fixture populates `obs_step` as `float(canvas_val) / 255.0` (coincidentally in [0,1]). A future bug where a raw unnormalized step index (0–40) is pushed would only be caught if any step value exceeds 1.0 — since `N_STROKES=40 < 255`, a raw integer step would not trigger the assertion. The canvas and step channels should be validated separately.

**Fix:**
```python
# Validate canvas channels and step channel independently
canvas_part = obs[:, :6, :, :]   # uint8 / 255
step_part   = obs[:, 6:7, :, :]  # normalized step

assert canvas_part.max() <= 1.0
assert canvas_part.min() >= 0.0
# Step channel should also be in [0, 1] by contract
assert step_part.max() <= 1.0
assert step_part.min() >= 0.0
```

---

## Info

### IN-01: Critic docstring says "value estimator V(s')" — reinforces the CR-01 naming confusion

**File:** `ddpg/agent.py:38-39`
**Issue:** The `critic` attribute is documented as "value estimator V(s')" which is the target network's role in the Bellman equation. The live critic should estimate `V(s)` (current state value). This ambiguity in the docstring contributed to the CR-01 pseudocode bug.

**Fix:** Update the docstring: `critic (Critic): Live critic network (V(s) — current state value estimator).` and `critic_target (Critic): Target critic (V(s') — used to bootstrap Bellman target, never trained directly).`

---

### IN-02: `import torch.nn as nn` in `agent.py` used only for a type annotation in `soft_update`

**File:** `ddpg/agent.py:3`
**Issue:** `nn` is imported at module level but used only as a type hint (`target: nn.Module`, `source: nn.Module`) in `soft_update`. This is a minor import weight issue.

**Fix:** Either use `torch.nn.Module` inline or use a string annotation `"torch.nn.Module"` and add `from __future__ import annotations`. Not urgent.

---

### IN-03: `import copy` is unused in `tests/test_agent.py`

**File:** `tests/test_agent.py:3`
**Issue:** `import copy` is present but never called in the test file. The deepcopy inside `DDPGAgent.__init__` is what gets tested, not a direct `copy.deepcopy` call in the test.

**Fix:** Remove `import copy` from `tests/test_agent.py`.

---

_Reviewed: 2026-06-10T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
