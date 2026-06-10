---
phase: 03-ddpg-models
plan: "02"
subsystem: ddpg
tags: [ddpg, replay-buffer, numpy, ring-buffer, uint8, memory, pytorch]

requires:
  - phase: 03-ddpg-models/03-01
    provides: Actor model (models/actor.py) — replay buffer is standalone but co-wave with actor

provides:
  - "ddpg/replay_buffer.py — ReplayBuffer numpy ring buffer, 200k capacity, uint8 canvas storage"
  - "tests/test_replay_buffer.py — 5 green tests for capacity, dtype, shape, range, GPU"

affects:
  - ddpg/agent.py (Plan 03-03 — agent push/sample calls)
  - training loop (Phase 4 — off-policy DDPG experience replay)

tech-stack:
  added: []
  patterns:
    - "uint8 canvas storage + scalar float32 step: reduces 200k-transition buffer from ~39 GB to ~9.84 GB"
    - "Step channel tiled lazily on sample() via .view(-1,1,1,1).expand(-1,1,H,W) — never stored as spatial"
    - "numpy ring buffer with O(1) push (ptr % capacity) and size = min(size+1, capacity)"

key-files:
  created:
    - ddpg/replay_buffer.py
    - tests/test_replay_buffer.py
  modified: []

key-decisions:
  - "D-15: Canvas stored as uint8 (6ch), step stored as float32 scalar — NOT tiled to (capacity,1,64,64)"
  - "D-17: sample() converts uint8 canvas to float32 [0,1] via .float().div(255.0)"
  - "D-14: Capacity = 200k transitions (~9.84 GB total), sufficient for MS COCO diversity"

patterns-established:
  - "ReplayBuffer.push takes (obs_canvas, obs_step, act, rew, next_canvas, next_step, done)"
  - "ReplayBuffer.sample returns exactly (obs, act, rew, next_obs, done) — five tensors"
  - "Test files use _SMALL_CAP=100 to avoid allocating full 200k buffer in CI"

requirements-completed: [DDPG-04]

duration: 3min
completed: "2026-06-10"
---

# Phase 03 Plan 02: Replay Buffer Summary

**Numpy ring buffer with uint8 canvas storage (200k transitions, ~9.84 GB), scalar float32 step channel tiled to (B,1,64,64) only at sample time**

## Performance

- **Duration:** 3 min
- **Started:** 2026-06-10T13:46:05Z
- **Completed:** 2026-06-10T13:48:39Z
- **Tasks:** 2 (TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments

- `ddpg/replay_buffer.py` implements a pre-allocated numpy ring buffer holding 200k transitions at ~9.84 GB RAM (vs ~39 GB naive float32), satisfying DDPG-04
- Canvas channels (3ch current + 3ch target) stored as `uint8`, converted to float32 `[0,1]` on sample via `.div(255.0)` per D-17
- Step channel stored as a 1-D float32 scalar array `(capacity,)` — tiled to `(B,1,H,W)` lazily in `sample()` only, avoiding 3.3 GB of pre-tiled spatial storage per buffer
- 5 tests green, 0 regressions (48/48)

## Task Commits

1. **Task 1: Failing buffer capacity/dtype/sample tests (RED)** - `acd0b46` (test)
2. **Task 2: Implement ddpg/replay_buffer.py (GREEN)** - `43cf0f3` (feat)

## Files Created/Modified

- `ddpg/replay_buffer.py` — ReplayBuffer class: `__init__`, `push`, `sample`, `__len__`; uint8 canvas + scalar step storage
- `tests/test_replay_buffer.py` — 5 tests: `test_capacity`, `test_uint8_storage`, `test_sample_shapes`, `test_sample_dtype_and_range`, `test_sample_gpu`

## Decisions Made

- uint8 canvas + scalar float32 step is the established storage pattern (D-15): keeps full 200k buffer within 31 GB RAM limit
- Step tiling happens only in `sample()` — never in `push()` or at rest in the arrays
- Test capacity uses `_SMALL_CAP=100` to avoid CI OOM; `REPLAY_BUFFER_CAPACITY` import kept for API contract documentation

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed ptr wrap assertion in `test_capacity`**
- **Found during:** Task 2 (GREEN phase — first test run)
- **Issue:** `test_capacity` asserted `buf.ptr == 10 % _SMALL_CAP` after pushing `5 + (capacity+10) = 115` transitions. The expected ptr is `115 % 100 = 15`, not `10 % 100 = 10`. The assertion used only the "over-capacity surplus" count (10), ignoring the initial 5 pushes.
- **Fix:** Changed assertion to `expected_ptr = (5 + _SMALL_CAP + 10) % _SMALL_CAP` (= 15)
- **Files modified:** `tests/test_replay_buffer.py`
- **Verification:** `pytest tests/test_replay_buffer.py -x -q` — 5 passed
- **Committed in:** `43cf0f3` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug in test assertion)
**Impact on plan:** Minor arithmetic error in test; fix is correct and verifiable. No scope creep.

## Issues Encountered

None beyond the test assertion bug above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `ddpg/replay_buffer.py` is fully standalone; `ddpg/agent.py` (Plan 03-03) can call `buf.push(...)` and `buf.sample(...)` directly
- Phase 4 training loop can import `ReplayBuffer` and use it immediately
- No blockers

## Self-Check

- [x] `ddpg/replay_buffer.py` exists: FOUND
- [x] `tests/test_replay_buffer.py` exists: FOUND
- [x] Commit `acd0b46` exists: FOUND
- [x] Commit `43cf0f3` exists: FOUND
- [x] `python -m pytest tests/ -x -q` exits 0: 48 passed

## Self-Check: PASSED

---
*Phase: 03-ddpg-models*
*Completed: 2026-06-10*
