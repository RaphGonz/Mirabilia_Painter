---
phase: 1
slug: foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-08
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.2 (installed) |
| **Config file** | `pyproject.toml` (Wave 0 creates it) |
| **Quick run command** | `pytest tests/ -x -q` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x -q`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** ~5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 0 | FOUND-01 | — | N/A | unit | `pytest tests/test_config.py -x` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 0 | FOUND-02 | — | N/A | unit | `pytest tests/test_palette.py -x` | ❌ W0 | ⬜ pending |
| 01-01-03 | 01 | 0 | FOUND-03 | — | NaN guard in oklab | unit | `pytest tests/test_renderer.py -x` | ❌ W0 | ⬜ pending |
| 01-01-04 | 01 | 0 | FOUND-01..03 | — | N/A | unit | `pytest tests/test_imports.py -x` | ❌ W0 | ⬜ pending |
| 01-02-01 | 02 | 1 | FOUND-03 | — | torch.no_grad prevents gradient attach | unit | `pytest tests/test_renderer.py::test_draw_no_autograd -x` | ❌ W0 | ⬜ pending |
| 01-02-02 | 02 | 1 | FOUND-03 | — | N/A | unit | `pytest tests/test_renderer.py::test_draw_full_canvas -x` | ❌ W0 | ⬜ pending |
| 01-02-03 | 02 | 1 | FOUND-03 | — | N/A | unit | `pytest tests/test_renderer.py::test_draw_subpixel_stroke_is_empty -x` | ❌ W0 | ⬜ pending |
| 01-02-04 | 02 | 1 | FOUND-03 | — | N/A | unit | `pytest tests/test_renderer.py::test_draw_gpu -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/__init__.py` — empty, marks tests as package
- [ ] `tests/test_config.py` — covers FOUND-01 (constants, types, values)
- [ ] `tests/test_palette.py` — covers FOUND-02 (project_color for all 3 colorspaces)
- [ ] `tests/test_renderer.py` — covers FOUND-03 (draw shape, edge cases, no-autograd, GPU)
- [ ] `tests/test_imports.py` — circular dependency check for all three modules
- [ ] `pyproject.toml` — pytest config with `testpaths = ["tests"]`
- [ ] `models/__init__.py` — empty, needed for future phases
- [ ] `ddpg/__init__.py` — empty, needed for future phases

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| draw() renders visually correct oriented rectangle | FOUND-03 | Shape correctness requires visual inspection | `python -c "import torch; from renderer import draw; import matplotlib.pyplot as plt; c=torch.zeros(3,64,64); p=torch.tensor([0.5,0.5,0.4,0.1,0.25,1.,0.,0.]); r=draw(c,p); plt.imshow(r.permute(1,2,0)); plt.savefig('/tmp/test_stroke.png')"` then inspect output |
| palette.py contains actual ~40 physical paint colors | FOUND-02 | Color accuracy is subjective | Manually verify _PALETTE_SRGB contains ~40 non-placeholder entries matching physical mixer |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
