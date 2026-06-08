---
phase: 01-foundation
plan: 01
subsystem: foundation
tags: [pytorch, pytest, colorspace, oklab, hsv, palette, config]

# Dependency graph
requires: []
provides:
  - config.py with six project-wide constants (IMG_SIZE=64, STROKE_DIM=8, STROKES_PER_STEP=5, N_STROKES=40, IMAGE_RANGE, PALETTE_COLORSPACE)
  - palette.py with _PALETTE_SRGB placeholder, PALETTE tensor, and project_color() for rgb/oklab/hsv colorspaces
  - pyproject.toml with pytest testpaths configuration
  - models/__init__.py and ddpg/__init__.py empty package init files
  - tests/__init__.py, tests/test_config.py, tests/test_palette.py (green)
  - tests/test_renderer.py scaffold (RED — renderer not yet implemented)
  - tests/test_imports.py circular-import order scaffold
affects: [02-renderer, 03-ddpg, 04-training, 05-eval]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "no-logic constants module: config.py has zero imports and zero logic"
    - "torch.cdist nearest-neighbor palette projection with .unsqueeze(0) on query"
    - ".clamp(min=0.0).pow(1/3) NaN guard in oklab LMS cube-root"
    - "branch-free HSV conversion via torch.where"
    - "flat absolute imports: from config import X (no relative or package-prefixed imports)"

key-files:
  created:
    - config.py
    - palette.py
    - pyproject.toml
    - models/__init__.py
    - ddpg/__init__.py
    - tests/__init__.py
    - tests/test_config.py
    - tests/test_palette.py
    - tests/test_renderer.py
    - tests/test_imports.py
  modified: []

key-decisions:
  - "PALETTE_COLORSPACE default is 'rgb' per RESEARCH open-question resolution (simplest baseline; oklab available for eval-time experimentation)"
  - "_PALETTE_SRGB is a 6-color placeholder; user must replace with actual ~40 physical paint mixer colors (D-05)"
  - "project_color raises ValueError for unsupported colorspace (explicit contract vs silent fallback)"
  - "tests/test_renderer.py is RED by design — renderer.py not created until Plan 02 (Nyquist scaffold)"

patterns-established:
  - "Pattern: no-logic constants module — config.py zero imports, zero logic"
  - "Pattern: torch.cdist shape contract — always .unsqueeze(0) single query to (1, D)"
  - "Pattern: clamp before pow(1/3) in oklab conversion"
  - "Pattern: flat absolute imports from project root"

requirements-completed: [FOUND-01, FOUND-02]

# Metrics
duration: 17min
completed: 2026-06-08
---

# Phase 1 Plan 01: Foundation Scaffold Summary

**config.py (6 constants, zero imports) + palette.py (oklab/hsv/rgb nearest-neighbor via torch.cdist) + Wave 0 test scaffold (4 test files, pyproject.toml, 3 package inits)**

## Performance

- **Duration:** 17 min
- **Started:** 2026-06-08T20:55:15Z
- **Completed:** 2026-06-08T21:12:00Z
- **Tasks:** 3/3
- **Files modified:** 10

## Accomplishments

- config.py exposes all six project constants with correct types and zero imports (FOUND-01 green)
- palette.py implements colorspace-aware nearest-neighbor projection for rgb, oklab, and hsv; black-oklab produces finite output (no NaN) via clamp guard (FOUND-02 green)
- Wave 0 test scaffold complete: pyproject.toml, 3 init files, 4 test files; tests/test_renderer.py is in RED state by design, ready for Plan 02

## Task Commits

Each task was committed atomically:

1. **Task 1: Create project scaffold and config.py** - `1baf754` (feat)
2. **Task 2: Implement palette.py and its test battery** - `95d4fbf` (feat)
3. **Task 3: Create the renderer test scaffold for Plan 02** - `60aba00` (feat)

## Files Created/Modified

- `config.py` — Six project-wide constants; zero imports, zero logic
- `palette.py` — _PALETTE_SRGB placeholder (6 colors), PALETTE tensor built at load, _srgb_to_linear, _linear_to_oklab (with NaN guard), _rgb_to_hsv, project_color()
- `pyproject.toml` — pytest testpaths = ["tests"]
- `models/__init__.py` — Empty package init for Phase 3+ actor/critic
- `ddpg/__init__.py` — Empty package init for Phase 3+ DDPG components
- `tests/__init__.py` — Empty package marker
- `tests/test_config.py` — 7 tests: all constant values and types (green)
- `tests/test_palette.py` — 8 tests: all 3 colorspaces, invalid colorspace ValueError, black-oklab NaN guard, tensor input (green)
- `tests/test_renderer.py` — 8 test scaffold functions for Plan 02 (RED until renderer.py exists)
- `tests/test_imports.py` — Circular-import order test (will be red until Plan 02)

## Decisions Made

- `PALETTE_COLORSPACE` default is `"rgb"` — simplest baseline; oklab available for eval-time experimentation without touching training code (RESEARCH open-question 3 resolution)
- `_PALETTE_SRGB` is a documented 6-color placeholder; user must replace with actual ~40 physical paint mixer colors divided by 255.0 (per D-05)
- `project_color` raises `ValueError` for unknown colorspace (explicit error contract; matches test_project_color_invalid_colorspace expectation)
- `tests/test_renderer.py` is RED by design — creating a stub renderer.py here would violate the Nyquist RED state requirement for Plan 02

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

- `palette.py:_PALETTE_SRGB` — 6-color placeholder (white, black, red, green, blue, yellow). This is intentional per RESEARCH open-question 1 and D-05: the user must enter the actual ~40 physical paint mixer colors. The `project_color` function is palette-size-agnostic and works correctly once filled in. This stub does not prevent FOUND-02 from being met (the API contract is satisfied; only the palette content is placeholder).

## Threat Flags

None — Phase 1 has no network, no user input, no file uploads. The oklab NaN guard (T-01-01) was implemented per the threat register.

## Issues Encountered

None.

## Next Phase Readiness

- Plan 02 (renderer.py) can now run against the test scaffold in tests/test_renderer.py — all 8 test functions exist and will fail at collection until renderer.py is created
- tests/test_config.py and tests/test_palette.py are green and stable
- config.py API is locked and will not change (every downstream module imports from it)
- User action before eval: replace `_PALETTE_SRGB` placeholder in palette.py with actual ~40 physical paint mixer colors

## Self-Check: PASSED

Files exist:
- config.py: FOUND
- palette.py: FOUND
- pyproject.toml: FOUND
- models/__init__.py: FOUND
- ddpg/__init__.py: FOUND
- tests/__init__.py: FOUND
- tests/test_config.py: FOUND
- tests/test_palette.py: FOUND
- tests/test_renderer.py: FOUND
- tests/test_imports.py: FOUND

Commits exist:
- 1baf754: FOUND
- 95d4fbf: FOUND
- 60aba00: FOUND

---
*Phase: 01-foundation*
*Completed: 2026-06-08*
