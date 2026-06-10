---
phase: 3
slug: ddpg-models
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-10
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pytest.ini` or `pyproject.toml` (existing) |
| **Quick run command** | `pytest tests/test_ddpg_models.py -x -q` |
| **Full suite command** | `pytest tests/ -x -q` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_ddpg_models.py -x -q`
- **After every plan wave:** Run `pytest tests/ -x -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 0 | DDPG-01 | — | N/A | unit | `pytest tests/test_ddpg_models.py::test_actor_shape -xq` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 1 | DDPG-01 | — | N/A | unit | `pytest tests/test_ddpg_models.py::test_actor_shape -xq` | ✅ | ⬜ pending |
| 03-02-01 | 02 | 0 | DDPG-02 | — | N/A | unit | `pytest tests/test_ddpg_models.py::test_critic_shape -xq` | ❌ W0 | ⬜ pending |
| 03-02-02 | 02 | 1 | DDPG-02 | — | N/A | unit | `pytest tests/test_ddpg_models.py::test_critic_shape -xq` | ✅ | ⬜ pending |
| 03-03-01 | 03 | 1 | DDPG-03 | — | N/A | unit | `pytest tests/test_ddpg_models.py::test_target_networks -xq` | ✅ | ⬜ pending |
| 03-04-01 | 04 | 1 | DDPG-04 | — | N/A | unit | `pytest tests/test_ddpg_models.py::test_replay_buffer -xq` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_ddpg_models.py` — stubs for DDPG-01 through DDPG-04 shape assertions
- [ ] `ddpg/__init__.py` — empty init for the ddpg sub-package

*Test file stubs for Wave 0: failing assertions that pass once actor/critic/agent/buffer are implemented.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| WN deepcopy safety for critic target | DDPG-03 | Requires runtime inspection of parametrization state | `import copy; from models.critic import Critic; c = Critic(); t = copy.deepcopy(c)` — must not raise |
| Buffer memory usage ≤ 12 GB | DDPG-04 | Requires OS-level memory monitoring | Run `python -c "from ddpg.replay_buffer import ReplayBuffer; b = ReplayBuffer(200000)"` and check process RSS |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
