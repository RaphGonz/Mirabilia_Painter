---
phase: 02-neural-renderer
plan: 02
subsystem: neural-renderer
tags: [neural-renderer, pretrain, supervised, checkpoint-pending]
dependency_graph:
  requires: [models/renderer.py, renderer.py, config.py]
  provides: [pretrain_renderer.py, load_frozen_renderer, param_norm, save_visual_gate, VISUAL_TEST_CASES]
  affects: [renderer.pkl, visual_gate.png, env.py]
tech_stack:
  added: []
  patterns: [supervised-pretraining, on-the-fly-datagen, extreme-param-biasing, freeze-load, visual-gate]
key_files:
  created:
    - pretrain_renderer.py
  modified: []
decisions:
  - "Tasks 1 and 2 co-committed (8868860): freeze/visual gate functions written alongside training loop in a single atomic write — both tasks' acceptance criteria fully met in one commit"
  - "matplotlib.use('Agg') placed before pyplot import to prevent Qt/Tk backend errors on headless machines"
  - "verbose= keyword omitted from ReduceLROnPlateau (Pitfall 2 — removed in PyTorch 2.x)"
  - "Target generation on CPU for 80% speedup over GPU loop (Pitfall 3 — 0.36s vs 0.80s at BS=1024)"
  - "Thin-stroke test cases use h/w=0.04 (above ~0.032 subpixel boundary) to ensure visible output (Pitfall 4)"
metrics:
  duration: "22 minutes"
  completed: "2026-06-09T18:52:12Z"
  tasks_completed: 2
  tasks_deferred: 1
  files_created: 1
  files_modified: 0
---

# Phase 2 Plan 2: pretrain_renderer.py Summary

**One-liner:** Standalone supervised training script for NeuralRenderer — on-the-fly data generation with 20% extreme-param biasing, MSE loss against hard rasterizer, freeze assertion, and visual gate figure export.

**Status: STOPPED AT CHECKPOINT — awaiting human training run (Task 3)**

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Implement pretrain_renderer.py (data gen + training loop + checkpoint save) | 8868860 | pretrain_renderer.py |
| 2 | Add freeze verification + visual gate functions (co-committed with Task 1) | 8868860 | pretrain_renderer.py |

## Task Deferred (Checkpoint Gate)

| Task | Name | Status | Reason |
|------|------|--------|--------|
| 3 | Run training on powerful machine — produce renderer.pkl and visual_gate.png | DEFERRED | autonomous=false — requires GPU training run by human |

## What Was Built

### pretrain_renderer.py — Complete Training Script (292 lines)

A standalone CLI script (`python pretrain_renderer.py`) that:

**Data generation functions:**
- `sample_uniform_batch(n)` — samples n stroke params uniformly from [0,1]^8
- `sample_extreme_batch(n)` — samples from extreme regions: thin-h, thin-w, tilted, full-canvas (4 quarters, D-06)
- `generate_targets(params)` — calls `draw(zeros_canvas, params[i])` in a CPU loop; returns `(B, 3, 64, 64)` float32
- `make_batch(batch_size)` — composes 80% uniform + 20% extreme params with targets; returns (params, targets) CPU tensors

**Training loop (in `main()`):**
- Device selection: CUDA if available, else CPU
- Pre-generates validation set (VAL_N=1000) once before training
- Trains for N_STEPS=976 steps (1M pairs at BS=1024)
- Per-step: make_batch on CPU, move to device, forward, MSE loss, backward, step
- Every VAL_EVERY=50 steps: val MSE under `torch.no_grad()`, ReduceLROnPlateau.step
- tqdm progress bar with train/val MSE and current LR
- Reports final val MSE; prints warning if >= 0.005 (RESEARCH.md A1)

**Checkpoint save:**
- `torch.save(R.state_dict(), 'renderer.pkl')` — state_dict only, never full module

**Post-save verification (in `main()` after save):**
- `load_frozen_renderer('renderer.pkl', cpu_device)` — loads with `weights_only=True` (T-02-PKL)
- `assert out.isfinite().all()` — NaN/Inf guard (T-02-NaN)
- `param_norm` assertion: `abs(norm_after - norm_before) < 1e-6` (T-02-FREEZE / REND-03)

**Visual gate:**
- `VISUAL_TEST_CASES` — 8 named (label, params) tuples: Thin H, Thin W, Tilted, Edge TL, Edge BR, Full canvas, Full+tilted, Extreme theta
- `save_visual_gate(R, path)` — 2-row matplotlib figure (hard rasterizer GT top, Neural R bottom) saved to `visual_gate.png`

**Freeze helpers (reusable by Phase 4 env.py):**
- `load_frozen_renderer(path, device)` — canonical freeze-load pattern: `load_state_dict` + `.eval()` + `requires_grad_(False)` (D-08)
- `param_norm(model)` — L2 norm over all parameters

## Security Mitigations Applied

| Threat | Mitigation | Location |
|--------|-----------|---------|
| T-02-PKL: pickle code execution via torch.load | `weights_only=True` in `load_frozen_renderer` | pretrain_renderer.py:82 |
| T-02-NaN: NaN propagation to Phase 4 RL loss | `assert out.isfinite().all()` after load | pretrain_renderer.py:240 |
| T-02-FREEZE: R parameters drifting during RL | `.eval()` + `requires_grad_(False)` + param_norm assertion `< 1e-6` | pretrain_renderer.py:85,246 |

## Deviations from Plan

### Tasks 1 and 2 Co-Committed

**Type:** Scope consolidation (non-breaking)
**Found during:** Writing pretrain_renderer.py
**Issue:** The plan calls for Task 2 to "extend" Task 1's output as a separate commit. However, all Task 2 functions (`load_frozen_renderer`, `param_norm`, `VISUAL_TEST_CASES`, `save_visual_gate`) were written together with Task 1's training loop functions in a single atomic write. This produces one commit (8868860) that satisfies all acceptance criteria for both tasks.
**Impact:** Zero — both tasks' verification commands pass, all acceptance criteria met. The single commit is cleaner than a two-commit split for content that was written together.
**Commit:** 8868860

## Checkpoint State

**Stopped at:** Task 3 — training run (autonomous=false, type=checkpoint:human-verify)

The script is complete and syntactically valid. It has NOT been run. The training run requires:
1. A machine with GPU (GTX 1660 Ti target: ~1h) or CPU (~6h)
2. Running `python pretrain_renderer.py` from the project root
3. Monitoring tqdm output for descending val MSE
4. Reporting final val MSE and `visual_gate.png` back to the SUMMARY

**Final val MSE:** NOT YET RECORDED (awaiting training run)
**renderer.pkl:** NOT YET PRODUCED
**visual_gate.png:** NOT YET PRODUCED

## Known Stubs

None — `pretrain_renderer.py` is a complete, correct implementation with no placeholder values or TODO stubs. The script cannot run until the training run task is executed, but all code is production-ready.

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes beyond what the plan's threat model covers.

## Self-Check: PASSED

- `pretrain_renderer.py` exists: FOUND
- Commit 8868860 (Task 1+2): FOUND
- All 5 required functions (Task 1): sample_uniform_batch, sample_extreme_batch, generate_targets, make_batch, main — CONFIRMED
- All 3 required functions (Task 2): load_frozen_renderer, param_norm, save_visual_gate — CONFIRMED
- VISUAL_TEST_CASES count: 8 — CONFIRMED
- matplotlib.use('Agg') before pyplot: CONFIRMED
- verbose=True absent: CONFIRMED
- weights_only=True present: CONFIRMED
- isfinite assertion present: CONFIRMED
- requires_grad_(False) present: CONFIRMED
- < 1e-6 freeze tolerance present: CONFIRMED
- make_batch(32) shape test: params (32,8), targets (32,3,64,64), range [0,1] — PASSED
- Full test suite (33 tests): PASSED (no regressions)
- Task 3 NOT executed: CONFIRMED (training deferred to human)
