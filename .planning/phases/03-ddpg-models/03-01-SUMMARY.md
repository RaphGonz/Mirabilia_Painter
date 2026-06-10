---
phase: 03-ddpg-models
plan: "01"
subsystem: models
tags: [ddpg, actor, resnet18, coordconv, batchnorm, pytorch, tdd]
dependency_graph:
  requires: []
  provides: [models/actor.py::CoordConv, models/actor.py::BasicBlock, models/actor.py::Actor]
  affects: [models/critic.py, ddpg/agent.py]
tech_stack:
  added: []
  patterns:
    - CoordConv with register_buffer for device-agnostic coord grids
    - ResNet18 BasicBlock with BatchNorm2d (expansion=1)
    - Actor ResNet18 backbone 4 stages [2,2,2,2] + sigmoid head
key_files:
  created:
    - models/actor.py
    - tests/test_actor.py
  modified: []
decisions:
  - "Actor output uses torch.sigmoid(self.fc(x)) for clean [0,1] range — no tanh+rescale"
  - "CoordConv inner Conv2d receives 9 channels (7 state + 2 coord grids)"
  - "CoordConv coord grids registered as buffers (.xx, .yy) for device-agnostic .to(device)"
  - "BasicBlock exported alongside CoordConv for reuse by models/critic.py (Plan 03-03)"
  - "No weight_norm in actor.py — that is the critic's concern per D-08"
metrics:
  duration: "~8 min"
  completed: "2026-06-10T13:43:07Z"
  tasks_completed: 2
  files_created: 2
---

# Phase 03 Plan 01: DDPG Actor (DDPG-01) Summary

**One-liner:** ResNet18+CoordConv actor CNN mapping (B,7,64,64) to (B,40) stroke params in [0,1] via sigmoid, with BatchNorm backbone and shared CoordConv/BasicBlock helpers.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Wave 0 scaffold — failing actor tests (RED) | 6990330 | tests/test_actor.py |
| 2 | Implement models/actor.py (GREEN) | 2ada112 | models/actor.py |

## Verification Results

```
python -m pytest tests/test_actor.py -x -q
5 passed in 2.26s

python -m pytest tests/ -x -q
43 passed in 2.06s  (no regression)

python -c "from models.actor import Actor, CoordConv, BasicBlock"
# Import OK
```

All acceptance criteria met:
- `Actor().eval()(torch.zeros(2,7,64,64)).shape == (2, 40)` ✓
- Output of `torch.rand(8,7,64,64)` has min >= 0.0 and max <= 1.0 ✓
- At least one `BatchNorm2d` in `Actor().modules()` ✓
- `Actor().coord_conv.conv.in_channels == 9` ✓
- `models/actor.py` contains `register_buffer('xx'` and `register_buffer('yy'` ✓
- `models/actor.py` contains `torch.sigmoid(self.fc` ✓
- `models/actor.py` does NOT contain `weight_norm` in functional code ✓

## Architecture Summary

```
obs (B, 7, 64, 64)
     |
[CoordConv(7->64, k=3, stride=2, pad=1)]  -- appends (x,y) buffers -> Conv2d(9,64,...)
     |
(B, 64, 32, 32)
     |
[Stage 1: 2x BasicBlock(64->64, stride=1)]   -> (B, 64,  32, 32)
[Stage 2: 2x BasicBlock(64->128, stride=2)]  -> (B, 128, 16, 16)
[Stage 3: 2x BasicBlock(128->256, stride=2)] -> (B, 256,  8,  8)
[Stage 4: 2x BasicBlock(256->512, stride=2)] -> (B, 512,  4,  4)
     |
[AdaptiveAvgPool2d(1,1) -> flatten(1)]       -> (B, 512)
     |
[Linear(512, 40) + sigmoid]                  -> (B, 40) in [0,1]
```

## Decisions Made

1. **sigmoid over tanh+rescale**: `torch.sigmoid(self.fc(x))` provides clean [0,1] range with no rescaling arithmetic. Per CONTEXT.md D-05.
2. **register_buffer for coord grids**: `self.register_buffer('xx', ...)` and `self.register_buffer('yy', ...)` ensure coordinate grids follow `.to(device)` without manual tracking. Mirrors `models/renderer.py` pattern.
3. **CoordConv and BasicBlock as shared helpers**: These classes are defined in `models/actor.py` and will be imported by `models/critic.py` in Plan 03-03. This avoids duplication — critic shares the same backbone building blocks with its own WN+TReLU variant (BasicBlockWN).
4. **No weight_norm in actor**: Actor uses BatchNorm per paper §3.4 and CONTEXT.md D-06. `weight_norm` mention in docstring is documentation only — zero functional weight_norm code.

## Deviations from Plan

None — plan executed exactly as written.

## TDD Gate Compliance

- RED gate: commit `6990330` (`test(03-01): add failing actor shape/range/BN tests (RED)`) — 5 tests failing with ModuleNotFoundError confirmed before implementation.
- GREEN gate: commit `2ada112` (`feat(03-01): implement CoordConv, BasicBlock, Actor in models/actor.py (GREEN)`) — 5 tests passing.
- No REFACTOR needed — implementation is clean as written.

## Known Stubs

None. All assertions are live (real module with working forward pass, not placeholder).

## Threat Flags

None. This plan introduces no new network endpoints, auth paths, file access patterns, or schema changes at trust boundaries. The only trust boundary is `training process → actor.forward` (internal float tensors only), which is mitigated by the sigmoid head bounding output to [0,1] — tested by `test_output_range`.

## Self-Check: PASSED

- [x] `models/actor.py` exists
- [x] `tests/test_actor.py` exists
- [x] Commit `6990330` exists (RED)
- [x] Commit `2ada112` exists (GREEN)
- [x] `python -m pytest tests/ -x -q` exits 0 (43 passed)
