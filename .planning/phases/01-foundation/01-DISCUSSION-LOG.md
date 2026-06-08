# Phase 1: Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-08
**Phase:** 1-Foundation
**Areas discussed:** N_STROKES value, Hard rasterizer implementation, Palette content, Project file layout

---

## N_STROKES Value

| Option | Description | Selected |
|--------|-------------|----------|
| 40 steps (200 strokes) | Short episode — faster to train, easier credit assignment. Good for 64×64. Matches paper's baseline. | ✓ |
| 100 steps (500 strokes) | Longer episode — more strokes for finer coverage, harder credit assignment. | |
| You decide | Claude picks a reasonable default. | |

**User's choice:** 40 steps (200 strokes)
**Notes:** N_STROKES=40 with k=5 gives 200 total strokes per episode at 64×64.

---

## Hard Rasterizer Implementation

| Option | Description | Selected |
|--------|-------------|----------|
| Pure PyTorch tensor ops | Rotation matrix + meshgrid pixel mask, GPU-compatible, scales to any resolution. | ✓ |
| cv2 now, rewrite later | Simpler to write correctly today, CPU-only, accepted technical debt. | |

**User's choice:** Pure PyTorch tensor ops

**Notes:** User initially asked for clarification on the rasterizer's role in training — confirmed that the hard rasterizer is NOT on the RL training hot path (only used for R pre-training data generation and final eval rendering). The neural renderer R handles all rendering during RL training. The PyTorch choice was driven by the user's stated intent to scale to larger resolutions (128×128, 256×256+) in future episodes — cv2 would require a rewrite at higher resolution during offline pre-training data generation.

**Sub-question — θ range:**

| Option | Description | Selected |
|--------|-------------|----------|
| [0, π] half-turn | Rectangles are 180°-symmetric — no redundant action representations. | ✓ |
| [0, 2π] full rotation | Matches [0,1] actor output directly but wastes action space. | |

**User's choice:** [0, π] — θ ∈ [0,1] maps to [0, π].

---

## Palette Content

| Option | Description | Selected |
|--------|-------------|----------|
| Actual colors now | Real RGB values from physical paint mixer entered in Phase 1. | ✓ |
| Placeholder for now | Small placeholder palette, fill in real colors before training. | |

**User's choice:** Actual colors now
**Notes:** Palette colors are from a physical paint mixer. Stored as float [0.0, 1.0] — divide 0–255 values by 255.

---

## Project File Layout

| Option | Description | Selected |
|--------|-------------|----------|
| Flat root | Top-level scripts at root, models/ and ddpg/ as subfolders. Simple imports. | ✓ |
| paint_ai/ package | Installable package structure, all files under paint_ai/. | |

**User's choice:** Flat root structure
**Notes:** Matches how REQUIREMENTS.md specifies file paths. Imports: `from config import IMG_SIZE`, `from models.renderer import NeuralRenderer`.

---

## Claude's Discretion

- `draw()` operates under `torch.no_grad()` — implementation detail not discussed but follows from FOUND-03's "no autograd graph" requirement.
- Float32 [0.0, 1.0] as the canvas tensor dtype — consistent with IMAGE_RANGE=(0.0, 1.0) in config.

## Deferred Ideas

- **Resolution scaling (128×128, 256×256+):** User confirmed intent to scale up in future episodes, mirroring the original paper's progression. Out of scope for this episode but drove the rasterizer implementation choice.
