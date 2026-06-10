---
phase: 03-ddpg-models
fixed_at: 2026-06-10T00:00:00Z
review_path: .planning/phases/03-ddpg-models/03-REVIEW.md
iteration: 1
findings_in_scope: 9
fixed: 8
skipped: 1
status: partial
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-06-10
**Source review:** `.planning/phases/03-ddpg-models/03-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 9 (3 Critical + 6 Warning)
- Fixed: 8
- Skipped: 1

---

## Fixed Issues

### CR-01: Correct Bellman pseudocode — `critic(obs)` not `critic(next_obs)`

**Files modified:** `ddpg/agent.py`
**Commit:** `aa557f9`
**Applied fix:** Changed `v_pred = critic(next_obs)` to `v_pred = critic(obs)` in the Phase 4
placeholder docstring. Also clarified the comment to `# (B, 1) V(s) — NOT next_obs`. The
`update_step` method is still a Phase 4 scaffold raising `NotImplementedError`; only the
comment was corrected.

---

### CR-02: Fix `soft_update` to use `state_dict` — weight-norm parametrization safe

**Files modified:** `ddpg/agent.py`
**Commit:** `f2975a7`
**Applied fix:** Replaced the `zip(target.parameters(), source.parameters())` loop that
mutated `.data` directly with a `state_dict()` / `load_state_dict()` approach. The old code
was a silent no-op for weight-norm parametrized modules because the parametrization hook
overwrites the derived `.weight` tensor on every forward pass from `weight_g` and `weight_v`.
`state_dict()` yields `weight_g`, `weight_v`, and all non-parametrized parameters as flat
keys so the blend propagates correctly. The docstring was updated to document the rationale.
Also updated the `critic` attribute docstring to read `V(s) — current state` (not `V(s')`).

---

### CR-03: Add `.contiguous()` after `expand()` in replay buffer

**Files modified:** `ddpg/replay_buffer.py`
**Commit:** `2c3a8e2`
**Applied fix:** Added `.contiguous()` after both `step.view(-1,1,1,1).expand(-1,1,H,W)` calls
(current obs and next obs). This materializes the stride-0 tensor before `torch.cat`, preventing
potential silent corruption on custom CUDA kernel paths and `torch.compile` usage.

---

### WR-01: Guard `sample()` against empty / undersized buffer

**Files modified:** `ddpg/replay_buffer.py`
**Commit:** `2c3a8e2`
**Applied fix:** Added a guard at the top of `sample()`:
```python
if self.size < batch_size:
    raise ValueError(
        f"Cannot sample {batch_size} transitions from buffer with only {self.size} stored."
    )
```
This gives a clear error instead of a cryptic `ValueError: low >= high` from numpy when the
buffer is empty, and prevents the silent biased-sampling case when `0 < size < batch_size`.

---

### WR-02: Cast `done` to `float32` in `sample()`

**Files modified:** `ddpg/replay_buffer.py`
**Commit:** `2c3a8e2`
**Applied fix:** Changed `torch.from_numpy(self.dones[idx]).to(device)` to
`torch.from_numpy(self.dones[idx]).float().to(device)`. Storage dtype stays `bool` (memory
efficient); only the returned tensor is cast to `float32`. The docstring return type was
updated from `done: (B,) bool` to `done: (B,) float32 — 0.0 (not done) or 1.0 (done)`.

---

### WR-03: CoordConv grids dynamic from input shape — remove hardcoded `IMG_SIZE`

**Files modified:** `models/actor.py`
**Commit:** `6f6cd1d`
**Applied fix:** Removed the registered buffers `self.xx` / `self.yy` computed at `__init__`
time with hardcoded `IMG_SIZE=64`. Replaced with on-the-fly grid generation inside `forward()`
using `B, _, H, W = x.shape` and `device=x.device`. This makes `CoordConv` resolution-
independent. `models/critic.py` imports `CoordConv` from `actor.py` so it automatically
benefits from the fix. The `IMG_SIZE` import was removed from `actor.py` (no longer used).

---

### WR-05: Fix tautological `test_soft_update` — perturb live critic before measuring

**Files modified:** `tests/test_agent.py`
**Commit:** `5b956ef`
**Applied fix:** Added a `with torch.no_grad()` block that adds `+1` to all live critic
parameters before capturing `p_before` and `p_source`. Previously, since `critic_target` is
a deepcopy of `critic` at init, `p_before == p_source`, making `expected == p_before` and the
assertion trivially pass even if `soft_update` was a no-op (which CR-02 confirmed it was).
Also removed the unused `import copy` (IN-03).

---

### WR-06: Validate canvas and step channels independently in `test_sample_dtype_and_range`

**Files modified:** `tests/test_replay_buffer.py`
**Commit:** `8b6f4d3`
**Applied fix:** Added separate assertions for `obs[:, :6, :, :]` (canvas channels) and
`obs[:, 6:7, :, :]` (step channel) after the existing aggregate bounds checks. A raw
unnormalized step index (0–40) would not trigger `obs.max() <= 1.0` since `N_STROKES=40 < 255`
and canvas values dominate. The per-channel check catches it directly.

---

## Skipped Issues

### WR-04: BatchNorm in actor backbone

**File:** `models/actor.py:55-56`
**Reason:** skipped — known-and-accepted design decision per CONTEXT.md D-02. The actor uses
a ResNet18 backbone which explicitly includes BatchNorm2d. This is documented in the
`Actor` class docstring as `"Design decision D-02: BN in actor is intentional (ResNet18
backbone)."` The reviewer's suggestion to replace BN with GroupNorm or WeightNorm+TReLU
would require re-training and is tracked as a future improvement, not a fix at this phase.
CLAUDE.md warns against BN at single-sample inference, but D-02 is the authoritative
decision for the baseline.

---

_Fixed: 2026-06-10_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
