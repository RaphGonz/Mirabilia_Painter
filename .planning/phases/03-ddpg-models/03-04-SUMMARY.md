---
phase: 03-ddpg-models
plan: "04"
subsystem: agent
tags: [ddpg, agent, target-networks, soft-update, deepcopy, pytorch, tdd]
dependency_graph:
  requires: ["03-01", "03-03"]
  provides: ["ddpg/agent.py::DDPGAgent", "ddpg/agent.py::soft_update", "DDPG-03"]
  affects: ["04-training-loop"]
tech_stack:
  added: []
  patterns:
    - "DDPGAgent: deepcopy'd target networks permanently in eval()+frozen via double-freeze"
    - "soft_update: @torch.no_grad() in-place tau-weighted blend (mul_ + add_)"
    - "update_step: NotImplementedError scaffold documenting Phase 4 Bellman/policy-gradient sequence"
key_files:
  created:
    - ddpg/agent.py
    - tests/test_agent.py
  modified: []
decisions:
  - "soft_update decorated with @torch.no_grad() and uses in-place mul_/add_ — no tensor allocation, no gradient flow to targets"
  - "Double-freeze for targets: .eval() disables BN/dropout behavior + requires_grad_(False) prevents gradient accumulation (mirrors pretrain_renderer.py::load_frozen_renderer)"
  - "update_step raises NotImplementedError with structured docstring documenting Phase 4 Bellman+actor-loss sequence (SoftRasterizer integration deferred)"
  - "deepcopy safety confirmed: parametrizations.weight_norm in critic (Plan 03-03) passes copy.deepcopy without RuntimeError"
metrics:
  duration: "~8 min"
  completed: "2026-06-10T14:01:48Z"
  tasks_completed: 2
  files_created: 2
  files_modified: 0
---

# Phase 03 Plan 04: DDPG Agent Scaffold (DDPG-03) Summary

**One-liner:** DDPGAgent scaffold wiring actor+critic (Plans 03-01/03-03) with deepcopy'd target networks permanently frozen in eval() mode, tau=0.005 soft update, and a NotImplementedError update_step placeholder marking the Phase 4 Bellman/policy-gradient boundary.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Wave 0 scaffold — failing agent deepcopy/eval/soft-update tests (RED) | 7937f88 | tests/test_agent.py |
| 2 | Implement ddpg/agent.py (DDPGAgent + soft_update) — GREEN | 6caf215 | ddpg/agent.py |

## What Was Built

### `ddpg/agent.py`

**`soft_update(target, source, tau)`** — Module-level function, `@torch.no_grad()`, in-place tau-weighted blend using `p_targ.data.mul_(1.0 - tau)` then `p_targ.data.add_(tau * p.data)`. No tensor allocation. No gradient flow.

**`class DDPGAgent`** — Constructor `DDPGAgent(device)`:
- `self.actor = Actor().to(device)` and `self.critic = Critic().to(device)` — live networks
- `self.actor_target = copy.deepcopy(self.actor)` and `self.critic_target = copy.deepcopy(self.critic)` — safe because critic uses `parametrizations.weight_norm`
- Double-freeze on both targets: `.eval()` + loop `requires_grad_(False)` on all params
- `self.actor_opt = Adam(actor.parameters(), lr=ACTOR_LR)` and `self.critic_opt = Adam(critic.parameters(), lr=CRITIC_LR)`
- `update_step(batch)` — raises `NotImplementedError` with structured docstring documenting the intended Phase 4 sequence (critic Bellman target, critic MSE + grad clip, actor loss through frozen SoftRasterizer, soft updates)

### `tests/test_agent.py`

Seven test functions covering DDPG-03:
- `test_targets_are_deepcopies`: identity check + value equality at init
- `test_target_eval_mode`: both targets `.training == False`
- `test_targets_frozen`: all params of both targets have `requires_grad == False`
- `test_soft_update`: `allclose((1-TAU)*p_before + TAU*p_source, p_after, atol=1e-6)`
- `test_critic_deepcopy_safe`: construction does not raise (proves WN API choice)
- `test_update_step_not_implemented`: `pytest.raises(NotImplementedError)`
- `test_agent_gpu`: CUDA device placement with `pytest.skip("No CUDA")` guard

## Verification Results

```
python -m pytest tests/test_agent.py -x -q
7 passed in 5.47s

python -m pytest tests/ -x -q
61 passed in 3.81s  (no regression)

python -c "from ddpg.agent import DDPGAgent, soft_update; import torch; DDPGAgent(torch.device('cpu'))"
# Construction OK; actor_target.training = False; critic_target.training = False
```

All acceptance criteria met:
- `soft_update` decorated `@torch.no_grad()`, uses `mul_(1` and `add_(` ✓
- `DDPGAgent` contains `copy.deepcopy(self.actor)` and `copy.deepcopy(self.critic)` ✓
- `DDPGAgent(torch.device('cpu'))` constructs without raising ✓
- `actor_target.training is False` and `critic_target.training is False` ✓
- All target params have `requires_grad == False` ✓
- After `soft_update(..., TAU)`, first target param equals `(1-TAU)*before + TAU*source` within atol=1e-6 ✓
- `agent.update_step(None)` raises `NotImplementedError` ✓
- `ddpg/agent.py` has no functional import of `SoftRasterizer` (docstring mentions only) ✓
- Full suite: 61 tests passed ✓

## Deviations from Plan

None — plan executed exactly as written.

## TDD Gate Compliance

- RED gate: commit `7937f88` (`test(03-04): add failing agent deepcopy/eval/soft-update tests (RED)`) — 7 tests failing with ModuleNotFoundError before implementation.
- GREEN gate: commit `6caf215` (`feat(03-04): implement DDPGAgent + soft_update in ddpg/agent.py (GREEN)`) — 7 tests passing, 61 total.
- No REFACTOR needed — implementation is clean as designed.

## Threat Surface Scan

| Threat | Mitigation | Status |
|--------|-----------|--------|
| T-03-07: Gradient flow into target networks | `@torch.no_grad()` on soft_update + `requires_grad_(False)` on all target params | MITIGATED |
| T-03-08: `copy.deepcopy(critic)` crash | `parametrizations.weight_norm` in critic (Plan 03-03); `test_critic_deepcopy_safe` asserts success | MITIGATED |
| T-03-SC: pip/conda installs | No new packages installed | ACCEPTED |

No new threat surfaces introduced.

## Known Stubs

`update_step()` raises `NotImplementedError` — this is an intentional Phase 3 scaffold, not a hidden stub. Phase 4 will implement the full Bellman update + actor policy gradient.

## Self-Check: PASSED

- [x] `ddpg/agent.py` exists with `def soft_update` and `class DDPGAgent`
- [x] `tests/test_agent.py` exists with 7 test functions
- [x] Commit `7937f88` exists (RED)
- [x] Commit `6caf215` exists (GREEN)
- [x] 61 tests pass, no regression
