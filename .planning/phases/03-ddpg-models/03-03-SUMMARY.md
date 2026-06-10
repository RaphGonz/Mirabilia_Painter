---
phase: 03-ddpg-models
plan: "03"
subsystem: critic
tags: [ddpg, critic, weight-norm, trelu, resnet18, pytorch, model-based]
dependency_graph:
  requires: ["03-01"]
  provides: ["models/critic.py", "tests/test_critic.py", "DDPG-02"]
  affects: ["03-04"]
tech_stack:
  added: []
  patterns:
    - "WeightNorm+TReLU critic backbone (paper §3.4) — WN via parametrizations API"
    - "CoordConv imported from actor.py (no duplication)"
    - "BasicBlockWN: parametrizations.weight_norm for deepcopy safety"
key_files:
  created:
    - models/critic.py
    - tests/test_critic.py
  modified: []
decisions:
  - "TReLU uses scalar alpha (torch.zeros(1)) per instance — not per-channel (RESEARCH Pitfall 5)"
  - "parametrizations.weight_norm used (not deprecated torch.nn.utils.weight_norm) — deepcopy-safe for Plan 03-04 agent"
  - "Downsample conv in _make_layer also wrapped with weight_norm (RESEARCH assumption A4)"
  - "stem_relu (TReLU) added after coord_conv output instead of F.relu inplace"
metrics:
  duration: "8 min"
  completed: "2026-06-10T13:55:28Z"
  tasks_completed: 2
  files_created: 2
  files_modified: 0
---

# Phase 03 Plan 03: Critic Model Summary

**One-liner:** Model-based DDPG critic using ResNet18+CoordConv+WeightNorm+TReLU mapping `(B,7,64,64)` rendered next-state to unbounded scalar V(s'), deepcopy-safe via `parametrizations.weight_norm`.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Wave 0 scaffold — failing critic tests (RED) | 282902b | tests/test_critic.py |
| 2 | Implement models/critic.py — TReLU + BasicBlockWN + Critic (GREEN) | ebb0c73 | models/critic.py |

## What Was Built

### `models/critic.py`

Three new classes:

**`TReLU(nn.Module)`** — Translated ReLU activation. `self.alpha = nn.Parameter(torch.zeros(1))` (scalar per instance). `forward: F.relu(x - self.alpha) + self.alpha` (non-inplace to preserve autograd graph integrity).

**`BasicBlockWN(nn.Module)`** — ResNet18 BasicBlock with `parametrizations.weight_norm` on conv1 and conv2, TReLU activations, no BatchNorm. `expansion = 1`. Accepts optional `downsample` (also wrapped with `weight_norm`).

**`Critic(nn.Module)`** — Model-based value network:
- `CoordConv(7, 64, stride=2)` stem → `(B, 64, 32, 32)` + TReLU
- 4 residual stages using `BasicBlockWN`: `[64→64, 64→128, 128→256, 256→512]`
- `AdaptiveAvgPool2d(1,1)` → flatten → `Linear(512, 1)`
- Output `(B, 1)` scalar V(s'), **no output activation** (unbounded)

`CoordConv` is imported from `models/actor.py` — not redefined.

### `tests/test_critic.py`

Six test functions covering DDPG-02:
- `test_critic_shape`: `(2,7,64,64)` → `(2,1)` shape assertion
- `test_critic_unbounded`: no NaN, no sigmoid/tanh clamp
- `test_critic_no_batchnorm`: zero BatchNorm modules (paper §3.4)
- `test_critic_uses_weight_norm`: `parametrize.is_parametrized` check
- `test_critic_input_is_image_not_concat`: `coord_conv.conv.in_channels == 9`
- `test_critic_gpu`: CUDA forward with `pytest.skip` guard

## Verification Results

```
python -m pytest tests/test_critic.py -x -q     → 6 passed
python -m pytest tests/ -x -q                   → 54 passed (no regression)
python -c "from models.critic import Critic, TReLU, BasicBlockWN"   → OK
python -c "import copy; from models.critic import Critic; copy.deepcopy(Critic())"  → OK
```

## Deviations from Plan

### Auto-fixed Issues

None — plan executed exactly as written.

**Minor implementation note (not a deviation):** The plan specifies `F.relu(self.coord_conv(x), inplace=True)` as the stem activation in the actor. For the critic, a `TReLU` instance is used as the stem activation (`self.stem_relu`) instead of a plain `F.relu` call, to maintain architectural consistency (all activations are TReLU in the critic backbone). This is consistent with the plan's action description and pattern specifications.

## Known Stubs

None — all outputs are wired. The critic fully computes `(B, 1)` from `(B, 7, 64, 64)` inputs.

## Threat Surface Scan

| Check | Result |
|-------|--------|
| T-03-05: deprecated weight_norm in critic.py | MITIGATED — `parametrizations.weight_norm` used; deprecated form absent (grep confirmed) |
| T-03-06: TReLU inplace op | MITIGATED — non-inplace `F.relu(x - self.alpha) + self.alpha` used |
| T-03-SC: pip/conda installs | ACCEPTED — no new packages installed |

No new threat surfaces introduced beyond those documented in the plan's threat model.

## Self-Check: PASSED

- [x] `models/critic.py` exists with `class TReLU`, `class BasicBlockWN`, `class Critic`
- [x] `tests/test_critic.py` exists with 6 test functions
- [x] Commit `282902b` exists (Task 1 RED)
- [x] Commit `ebb0c73` exists (Task 2 GREEN)
- [x] `parametrizations.weight_norm` used; deprecated form absent
- [x] `CoordConv` imported from `models/actor.py`, not redefined
- [x] `coord_conv.conv.in_channels == 9` confirmed
- [x] `TReLU.alpha.numel() == 1` confirmed (scalar, not per-channel)
- [x] `copy.deepcopy(Critic())` succeeds
- [x] 54 tests pass, no regression
