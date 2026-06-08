<!-- GSD:project-start source:PROJECT.md -->

## Project

**Paint AI — Mirabilia Épisode 1**

Réimplémentation du papier "Learning to Paint" (DDPG + renderer neuronal différentiable).
Un agent RL apprend à peindre une image cible en posant des traits rectangulaires opaques sur une toile 64×64, guidé par un signal L2 incrémental.
Projet documenté publiquement dans la série de contenu **Mirabilia** — chaque épisode trace la progression de la baseline jusqu'aux évolutions.

**Core Value:** L'agent produit un timelapse demo-able : on peut filmer l'IA qui peint une image cible trait par trait, de façon reconnaissable.

### Constraints

- **Tech stack**: PyTorch + CUDA (GPU NVIDIA local)
- **Image size**: 64×64 — itération rapide, montée en résolution dans épisodes futurs
- **Stroke params**: `(cx, cy, w, h, θ, r, g, b)` — STROKE_DIM=8
- **Bundle size**: k=5 traits/step — décision verrouillée baseline
- **Reward**: L2 incrémental uniquement — pas de WGAN avant que la baseline fonctionne
- **Palette**: ~40 couleurs, projection nearest-neighbor RGB — décision verrouillée baseline
- **N_STROKES**: fixe et élevé — seuil d'arrêt a posteriori, pas de signal stop appris

<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->

## Technology Stack

## Recommended Stack

### Core ML Framework

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Python | 3.11 | Runtime | 3.11 is the sweet spot: fully supported by PyTorch 2.x, mature ecosystem, faster than 3.10, less churn than 3.12/3.13. Avoid 3.13+ — some packages lag. |
| PyTorch | 2.7.0 | Core tensor ops, autograd, model training | Latest stable on pytorch.org as of 2026-06 (PyPI shows 2.12 which is the build system version; pytorch.org installer stable is 2.7). CUDA 12.6 or 12.8 bundle included — no separate CUDA toolkit install needed. |
| torchvision | 0.22.x | Image I/O utilities, transforms | Paired with PyTorch 2.7. Use only for `transforms.ToTensor()`, `transforms.Normalize()`, and `save_image()`. Do NOT use its deprecated video I/O. |
| CUDA | 12.6 or 12.8 | GPU acceleration | Bundled in PyTorch wheel. Pick 12.6 for widest NVIDIA driver compatibility on Windows; 12.8 if driver is recent (>=555). |

### DDPG Implementation

| Component | Pattern | Notes |
|-----------|---------|-------|
| Actor | `nn.Module` CNN → `tanh` output in [-1,1], rescaled to [0,1] | State is (canvas+target) stacked: 6×64×64. Output: 40 dims (k=5 × STROKE_DIM=8). Use `sigmoid` instead of `tanh+rescale` for cleaner [0,1] range. |
| Critic | `nn.Module` CNN extracts obs embedding, then concat with action, pass to MLP | Concatenate action AFTER CNN feature extraction, NOT as input channels. Q-value: scalar output, no activation. |
| Target networks | `copy.deepcopy(actor)` and `copy.deepcopy(critic)`, weights updated via soft update only | `for p, p_targ in zip(net.parameters(), targ.parameters()): p_targ.data.mul_(1-tau).add_(tau * p.data)` |
| Replay buffer | Pure Python `collections.deque` or numpy ring buffer | At 64×64 RGB with k=5, transitions are small. A 1M-transition buffer is ~3 GB if stored naively as float32 tensors. Store as `uint8` canvas images + float32 params to keep under 1 GB. |
| Exploration noise | Gaussian noise `N(0, σ)` decaying over training | OU noise (Ornstein-Uhlenbeck) adds complexity without meaningful benefit for painting tasks where temporal correlation in noise is not essential. Gaussian with σ annealed from 0.3 → 0.05 is simpler and works. |
| Optimizer | `torch.optim.Adam` | Actor lr=1e-4, Critic lr=1e-3. Standard DDPG values. |
| Gradient clipping | `torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)` | Apply to critic only — critic Q-values can explode early in training. |

### Neural Renderer

| Component | Choice | Why |
|-----------|--------|-----|
| Architecture | Small CNN decoder with upsampling (not MLP) | MLP for 64×64=4096 pixel output has poor spatial inductive bias. A 3–4 layer conv decoder with bilinear upsampling generalizes better to stroke geometry. |
| Output | Single stroke image (3 channels RGB) + optional alpha mask | Keep composition out of R. R outputs the stroke alone; compositing onto canvas is done in `env.py` via alpha blending or hard overwrite. |
| Loss | MSE (L2) against hard rasterizer output | Pixel-wise MSE is sufficient for supervised pre-training of R. No perceptual loss needed at 64×64. |
| Optimizer | Adam, lr=1e-3, with ReduceLROnPlateau | Standard supervised training setup. |
| Training data | Random stroke params sampled uniformly from [0,1]^8, rendered with hard rasterizer | Generate on-the-fly during pre-training — no static dataset needed. Sample ~500K–1M (stroke, image) pairs. |
| Batch norm | No | BN interacts poorly with single-sample inference at test time. Use layer norm or group norm if normalization is needed, but likely unnecessary for this task. |
| Input | stroke params vector, 8 dims | Expand to spatial via learned upsampling from 8-dim latent, not pixel grid. |
| Freezing | `model.eval()` + `for p in R.parameters(): p.requires_grad_(False)` | Both are necessary: `eval()` disables dropout/BN behavior, `requires_grad_(False)` saves memory and prevents accidental gradient flow. |

### Visualization & Logging

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| tensorboard | 2.x (latest) | Training curves (reward, loss, Q-values) | Best tool for RL training monitoring. Integrates via `torch.utils.tensorboard.SummaryWriter`. Zero config, stays local, no account needed. |
| matplotlib | 3.9.x | Offline plots, side-by-side canvas vs target | Use for saving comparison figures during eval. `plt.savefig()` to disk, not interactive display during training. |
| imageio | 2.x | Timelapse export (GIF and MP4) | `imageio.get_writer('timelapse.gif', mode='I')` + `writer.append_data(frame)` is the simplest path. For MP4, use `imageio-ffmpeg` plugin. |
| opencv-python | 4.x | Hard rasterizer pixel ops, BGR/RGB conversion | Use `cv2` only for the hard rasterizer's rectangle drawing if needed. Avoid for general image I/O — stick to `torchvision.utils.save_image` for tensor saves. |
| numpy | 1.26.x or 2.0.x | Array ops, replay buffer storage | Required by imageio and cv2. Use `np.uint8` for canvas storage in replay buffer to save memory. |
| tqdm | 4.x | Training loop progress bar | One-liner: `for step in tqdm(range(MAX_STEPS))`. Essential for long training runs. |

### Dev Environment

| Tool | Version | Purpose | Why |
|------|---------|---------|-----|
| conda / mamba | latest | Environment isolation | Conda handles CUDA-aware PyTorch install on Windows better than pure pip venvs. Use `mamba` (drop-in, faster solver) if available. |
| Jupyter Lab | 4.x | Interactive prototyping for renderer pretraining | Run renderer training and visualize output inline before wiring into the RL loop. Do NOT use Jupyter for the main DDPG training loop — use a plain `.py` script. |
| VS Code | latest | Primary editor | Python extension + Pylance for type checking. CUDA debugger via Nsight if needed. |
| git | 2.x | Version control | Checkpoint model weights as `.pt` files tracked or gitignored depending on size. Use `.gitignore` to exclude checkpoints > 100 MB. |

## What NOT to Use

### Stable-Baselines3

### TorchRL

### RLlib (Ray)

### OpenAI Gym (legacy, pre-1.0)

### TensorFlow / JAX

### Weights & Biases (wandb)

### MLP-only Neural Renderer

### PIL/Pillow as primary image handler

### autograd through the hard rasterizer

## Confidence Notes

| Recommendation | Confidence | Basis |
|---------------|------------|-------|
| PyTorch 2.7.0, Python 3.11 | HIGH | pytorch.org stable installer, PyPI verified |
| CUDA 12.6 bundle (no separate toolkit) | HIGH | Official PyTorch docs confirm bundled CUDA |
| DDPG from scratch (not SB3/TorchRL) | HIGH | Verified against SB3 docs, TorchRL docs, and the specific constraints of this project (custom renderer, non-standard obs) |
| CNN decoder for renderer R (not MLP) | HIGH | Spatial inductive bias is a well-established principle; confirmed by community discussion on painter DDPG architectures |
| Gaussian noise over OU noise | MEDIUM | OU is standard in original DDPG paper; Gaussian is simpler and adequate — based on community practice, not a formal comparison for painting tasks |
| TensorBoard over wandb | HIGH | Local dev constraint, zero-dependency preference clearly favors TensorBoard |
| imageio for timelapse | HIGH | imageio docs verified; `imageio-ffmpeg` plugin confirmed for MP4 output |
| Replay buffer as custom numpy ring buffer | HIGH | Standard practice; SB3 and TorchRL both use similar patterns internally |
| TAU=0.005 (not 0.001) | MEDIUM | 0.001 is original DDPG paper value; 0.005 is SB3 default and empirically faster for dense-reward tasks like painting. Tunable. |
| No batch norm in renderer | MEDIUM | BN + single-sample inference is a known pitfall (training vs eval mode); training with large batches mitigates but avoiding BN is safer for this frozen-at-inference use case |

## Sources

- PyTorch stable version: https://pytorch.org/get-started/locally/
- PyTorch PyPI page: https://pypi.org/project/torch/
- PyTorch 2.8 release blog: https://pytorch.org/blog/pytorch-2-8/
- TorchRL stable docs (0.13): https://docs.pytorch.org/rl/stable/index.html
- TorchRL DDPG reference implementation: https://github.com/pytorch/rl/blob/main/sota-implementations/ddpg/ddpg.py
- TorchRL ReplayBuffer docs: https://docs.pytorch.org/rl/stable/reference/data_replaybuffers.html
- Stable-Baselines3 DDPG docs: https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html
- SB3 custom environments: https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
- PyTorch DDPG with CNN discussion: https://discuss.pytorch.org/t/ddpg-agent-with-convolutional-layers-for-feature-extraction/154006
- Learning to Paint original repo: https://github.com/hzwer/ICCV2019-LearningToPaint
- imageio usage examples: https://imageio.readthedocs.io/en/stable/examples.html
- torchvision write_video (deprecated note): https://docs.pytorch.org/vision/main/generated/torchvision.io.write_video.html
- Gymnasium custom env: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
