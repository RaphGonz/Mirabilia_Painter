# Pitfalls Research — Paint AI

**Domain:** DDPG-based computational painting with differentiable neural renderer
**Researched:** 2026-06-08
**Reference paper:** "Learning to Paint With Model-Based Deep Reinforcement Learning" (Huang et al., ICCV 2019)

---

## Critical Pitfalls (can kill the project)

### CP-1: Gradient Flow Through the Frozen Renderer R

**What goes wrong:** During DDPG actor updates, the gradient of the actor loss passes through R (the neural renderer) to reach the actor parameters. If R is accidentally left in training mode (gradients enabled), two bad things happen simultaneously: (1) R's weights drift from the pre-trained optimum, silently corrupting the reward signal; (2) gradients accumulate inside R's graph across batches, causing memory bloat and eventually OOM or NaN loss.

**Why it happens:** PyTorch's autograd tracks all operations by default. Freezing a network requires both `requires_grad_(False)` on parameters AND wrapping inference calls in `torch.no_grad()` (or storing R's output as a detached tensor before it enters the composition pipeline). Doing only one of these leaves a silent path open.

**Consequences:** R slowly unlearns its stroke representation. The RL agent then trains against a renderer that is shifting under it — indistinguishable from "DDPG not converging" in a loss curve. Usually only caught by visually inspecting R's outputs after thousands of steps.

**Prevention:**
- After pre-training R, call `R.requires_grad_(False)` and immediately verify with `any(p.requires_grad for p in R.parameters())`.
- In `env.py`'s `step()`, wrap the call to R inside `with torch.no_grad():`.
- Add an assertion in the training loop that checks R's parameter norm hasn't changed from its post-pretrain checkpoint — a single line that catches accidental unfreeze.

**Warning signs:** R output images gradually become blurry or shift; actor loss curve is flat but critic loss diverges.

**Phase:** Renderer pre-training (validate freeze) + RL training setup (assert freeze).

---

### CP-2: DDPG Critic Divergence (Q-Value Explosion)

**What goes wrong:** The critic's Q-values grow unbounded (common values: 1e4–1e6 within a few thousand steps), making the actor gradient meaningless. Once Q explodes, soft-update copies the corrupted values into the target network, and the system never recovers without a full restart.

**Why it happens:** DDPG uses a bootstrapped TD target `y = r + γ·Q_target(s', π(s'))`. If the critic overestimates Q even slightly, the target becomes inflated, which inflates the next estimate, which inflates the next target — a positive feedback loop. High-dimensional action spaces (40 dims for k=5 bundles) amplify this because the actor can find action directions that exploit Q-function inaccuracies more easily.

**Consequences:** Total training failure. Loss goes to NaN. Policy outputs constant max-value actions (saturated tanh).

**Prevention:**
- Log `Q_mean` and `Q_max` every 100 steps. If `Q_max > 10 * expected_max_return`, stop and investigate.
- Expected max return for L2 incremental reward: upper-bounded by the initial L2 distance to target (typically ~0.3–0.8 for normalized images). Q should stay in this range.
- Apply gradient clipping to the critic: `torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)`.
- Keep τ (soft update) at 0.005 or smaller for this task. Values above 0.01 increase divergence risk.
- If Q explosion persists, the drop-in fix is TD3 (twin critics, target policy smoothing) — already listed in the project's evolution path.

**Warning signs:** Critic loss `> 100` after warm-up; actor always outputs `[1.0, 1.0, ..., 1.0]` or `[0.0, 0.0, ..., 0.0]`; reward curve is flat after initial rise.

**Phase:** RL training loop (monitoring), first sign of instability.

---

### CP-3: Neural Renderer R Validates on Train Distribution Only

**What goes wrong:** R is pre-trained on randomly sampled stroke parameters and achieves near-zero loss. However, when the RL agent starts training, it will issue stroke parameters outside the high-density training region (e.g., very small strokes, near-zero width, extreme angles). R produces garbage outputs for these OOD inputs, but the agent receives a gradient signal as if they were valid — steering the policy into a broken region of parameter space.

**Why it happens:** Uniform sampling across `[0,1]^8` covers the space evenly but visually important edge cases (thin strokes: h~0.02, rotated 45°; full-canvas strokes: w=h=1.0; near-black colors: r=g=b~0.02) are very sparse in a uniform distribution. A 500k-step training run with batchsize 64 sees roughly 32M random strokes, but the 8D parameter space has infinite capacity for rare combinations.

**Consequences:** The RL agent learns to exploit R's blind spots. Canvas images look valid during training (R outputs something) but the hard rasterizer at eval time produces completely different strokes. The train/inference visual gap becomes severe.

**Prevention:**
- Before any RL training, run a coverage audit: sample 10k strokes uniformly, render via hard rasterizer AND R, compute per-pixel L2. Plot the distribution. Tails (L2 > 0.05) indicate underrepresented regions.
- Add biased sampling to R's pre-training: 20% of batches should use extreme parameter values (w or h < 0.05, w or h > 0.95, θ near 0/45/90°, near-black colors). This costs nothing and prevents the most common failure mode.
- Validate R's quality with a held-out set of strokes that includes purposefully extreme parameters.

**Warning signs:** Eval timelapse looks "melted" or has ghost strokes; large gap between the loss-curve canvas (rendered via R) and the final hard-rasterizer replay; agent prefers very small or very large strokes in training but not in eval.

**Phase:** Renderer pre-training (coverage), before RL begins.

---

### CP-4: Wrong Critic Architecture — Q(s,a) vs V(s')

**What goes wrong:** Standard DDPG tutorials implement the critic as `Q(s, a) -> scalar`. The Huang et al. paper uses a model-based variant where the critic takes the **next state** `s'` (after applying the stroke bundle through R) as input, not `(s, a)`. Implementing the standard architecture instead means the critic never sees the outcome of the action — it must infer paint quality from action parameters directly, which is far harder and converges much more slowly.

**Why it happens:** Most DDPG references (Spinning Up, Stable Baselines, tutorials) show the standard `Q(s,a)` form. The paper's model-based twist is described briefly and easy to miss.

**Consequences:** Training may appear to progress (loss decreases) but painting quality plateaus far below what the paper achieves. The critic cannot learn a useful value function from raw stroke params and a canvas, without seeing the rendered result.

**Prevention:**
- In `critic.py`: input is `s_{t+1}` = the next canvas state after applying the stroke bundle via R. Do NOT concatenate the raw action params with the current state.
- Explicitly comment this deviation from standard DDPG in the code.
- Verify by checking: does the critic receive a 6×64×64 image tensor (next canvas + target) as input? If it receives action params (floats), the architecture is wrong.

**Warning signs:** Critic converges quickly on easy images but fails on complex ones; the actor outputs strokes that look "reasonable" but never refine detail; overall performance plateau at ~40% reconstruction quality.

**Phase:** Models and DDPG setup (architectural choice).

---

### CP-5: Bundle Reward Assigned to Wrong State Transition

**What goes wrong:** With k=5 strokes per step, the environment applies 5 strokes sequentially to the canvas but the RL transition is stored as `(s_t, actions_bundle, reward, s_{t+k})`. If the bundle is composed incorrectly (e.g., all 5 strokes applied to `s_t` independently and then averaged, rather than sequentially), the stored next-state `s'` does not correspond to what the reward measures. The critic learns a broken value function.

**Why it happens:** When implementing the bundle, it's tempting to parallelize: render all k strokes against the current canvas and stack results. The correct implementation applies stroke 1, updates canvas, applies stroke 2, updates canvas, etc. The sequential version is slower but the parallel version is physically wrong (k strokes all painted "on top of" the same base canvas ignores occlusion).

**Consequences:** Strokes within a bundle compete rather than cooperate. Early strokes get "overwritten" visually but their contribution to the reward is lost. The agent learns to put all color information in stroke 5 (the last one). Timelapse looks like a single-stroke update per step.

**Prevention:**
- In `env.py step()`: use a loop, not a batched parallel call, to apply the k strokes in sequence.
- Assert that `canvas` changes after each individual stroke application within the loop (single-step sanity check).
- Log per-stroke L2 improvement during a debug run to verify each stroke contributes incrementally.

**Warning signs:** Visual inspection shows only the last stroke of each bundle having any effect; reward histogram shows a bimodal distribution (step-function gains).

**Phase:** Environment implementation.

---

## Common Mistakes (slow you down)

### CM-1: Reward Magnitude Not Calibrated

**What goes wrong:** L2 incremental reward `r_t = L2(prev_canvas, target) - L2(new_canvas, target)` can range from approximately -0.01 to +0.05 per bundle for a 64×64 image, depending on whether images are normalized to [0,1] or [0,255]. If images are stored as uint8 and the reward is computed in raw pixel units, reward values are ~65000× larger. The Q-value target explodes immediately.

**Prevention:**
- Normalize canvas and target to [0,1] float before all reward computation. Keep this normalization consistent throughout the pipeline.
- After first 100 training steps, verify `abs(reward).mean() < 0.1`. Values > 1.0 indicate a normalization bug.
- Log reward stats (mean, std, min, max) for the first 1000 steps. Any value outside [-0.5, +0.5] for normalized images deserves investigation.

**Warning signs:** Critic loss jumps to > 1000 in the first few hundred steps; Q values are in the thousands.

**Phase:** Reward function implementation, early training.

---

### CM-2: Action Space Not Normalized to [0,1]

**What goes wrong:** The actor outputs raw stroke parameters. Geometric params `(cx, cy, w, h)` naturally map to [0,1] (fraction of image size), but color `(r, g, b)` and angle `θ` may be treated differently. If `θ` is in radians `[0, π]` or color is in `[0, 255]`, the actor's output range for different parameters is inconsistent. The optimizer treats a gradient of 0.01 on `cx` and 0.01 on `θ` as equivalent, but they represent very different changes.

**Prevention:**
- Normalize ALL action dimensions to [0,1] in the actor's output (via sigmoid or hard clamp). Scale back to physical units only inside the hard rasterizer and R's input preprocessing.
- `config.py` should define `ACTION_LOW = [0]*8` and `ACTION_HIGH = [1]*8` as the canonical contract. Any module that deviates is a bug.

**Warning signs:** Agent learns color well but struggles with rotation; gradient magnitudes for different action dimensions differ by > 100×.

**Phase:** Models/actor setup.

---

### CM-3: Exploration Noise Swamped or Too Weak

**What goes wrong:** DDPG uses additive noise (Ornstein-Uhlenbeck or Gaussian) for exploration. For a 40-dimensional action space (k=5, 8 params each), noise added to one "good" action dimension is diluted by 39 other dimensions. If noise σ is too small, the agent doesn't explore; if too large, every step is destructive (negative reward), filling the replay buffer with bad experiences and slowing value learning.

Additionally, once the actor saturates (outputs near 0 or 1 due to tanh squashing), additive noise has almost no effect — the noise is added after the tanh, but the gradient for exploration that pushes the pre-tanh activations away from saturation does not exist in vanilla DDPG. The agent gets "stuck" in a region of action space.

**Prevention:**
- Start σ at 0.1 and schedule it to decay to 0.01 over training. Log the fraction of actions that are clipped at boundary values (near 0 or 1); if > 30%, reduce σ or add a pre-tanh noise option.
- Alternatively, use parameter-space noise instead of action-space noise — it's more robust to tanh saturation.
- Log `action.std()` per episode; it should stay above 0.05 throughout warm-up.

**Warning signs:** Reward improves quickly then plateaus; action histograms show spikes at 0.0 or 1.0; agent paints with only 1-2 distinct colors.

**Phase:** DDPG training loop.

---

### CM-4: Replay Buffer Memory Overflow

**What goes wrong:** Each transition in the replay buffer stores two full canvas states (current and next) as 6×64×64 float32 tensors. A buffer of 1M transitions would require approximately `1,000,000 × 2 × 6 × 64 × 64 × 4 bytes = ~6 GB RAM`, before accounting for actions, rewards, and done flags. On a system with 16 GB RAM and an active GPU, this causes OOM at buffer initialization.

**Prevention:**
- Store canvas states as `uint8` (values 0–255) and convert to float32 only at sample time. This reduces per-transition storage 4×, to ~1.5 GB for 1M transitions.
- Alternatively, start with a buffer of 100k–200k transitions (typical painting tasks don't require 1M; the state distribution is already rich due to varying target images).
- The original repo had this OOM issue — pre-loading all images into RAM simultaneously caused kills on low-memory machines (GitHub issue #23).

**Warning signs:** Python OOM kill at buffer initialization; training process is killed with no error message.

**Phase:** Replay buffer implementation.

---

### CM-5: Checkerboard Artifacts in Neural Renderer Output

**What goes wrong:** R uses sub-pixel convolution (PixelShuffle) for upsampling, which is correct in principle. However, if the convolutional kernel sizes before each PixelShuffle are not carefully chosen (specifically, kernel size must be divisible by the upscaling factor), the resulting stroke images contain a visible checkerboard grid. These artifacts propagate into the canvas and corrupt the reward signal — the agent gets penalized for artifacts it didn't "intend."

**Prevention:**
- When implementing R's architecture, ensure each conv layer before PixelShuffle(r) uses kernel_size divisible by r (e.g., kernel_size=4 before PixelShuffle(2)).
- During R's validation step, visually inspect the output on 20+ random strokes before any RL training. Checkerboard patterns are immediately visible.
- If artifacts appear, replace PixelShuffle with bilinear upsampling + conv, which is less elegant but artifact-free.

**Warning signs:** Grid-like pattern visible in R's output images at the pixel level; the hard rasterizer shows clean strokes but R's strokes have periodic banding.

**Phase:** Renderer pre-training, visual validation step.

---

### CM-6: Step Counter Not Included in State

**What goes wrong:** The state fed to the actor is `(canvas, target)` — 6 channels. The remaining step budget is not included. Without it, the actor cannot learn to "finish" the painting (place fine strokes late in the episode vs. broad strokes early). The agent treats every step identically, leading to a tendency to keep laying down large strokes even at step N-1.

The original paper explicitly includes the step number `t` in the state.

**Prevention:**
- Add a scalar channel or a broadcast scalar to the state: either tile a (1×64×64) map filled with `(current_step / N_STROKES)` and concatenate as a 7th channel, or include it as a separate scalar input to the actor's fully-connected head.
- Verify the step counter is normalized to [0,1]. Raw step numbers (e.g., 0–100) fed to a CNN create non-stationary inputs.

**Warning signs:** Agent paints fine details in step 1 and coarse blocks in step 50; timelapse looks "random" rather than progressive.

**Phase:** Environment and actor setup.

---

### CM-7: Renderer R Receives Raw Action Params From Actor Without Preprocessing

**What goes wrong:** The actor outputs values in [0,1]. The renderer R was pre-trained to accept params in the same [0,1] range. However, if the hard rasterizer uses different conventions (e.g., `cx, cy` in pixel coordinates 0–64, `θ` in degrees 0–180), and R is pre-trained against the hard rasterizer's coordinate space without normalizing, then R and the actor speak different "languages."

**Prevention:**
- Define a single canonical parameterization in `config.py` (all values [0,1]) and make all modules (actor, R, hard rasterizer) use this. The hard rasterizer rescales internally.
- Write a unit test: `hard_rasterizer(params)` and `R(params)` on the same input should produce visually similar strokes (same location, size, orientation). Run this test before starting RL.

**Warning signs:** R renders strokes in the center of the image regardless of `cx, cy`; stroke sizes are all uniform despite varying `w, h`.

**Phase:** Integration between renderer pre-training and RL environment.

---

## Subtleties (easy to miss)

### S-1: Two Distinct Losses for R — MSE and Perceptual

**What goes wrong:** Training R with pixel-level MSE loss alone tends to produce "average" stroke shapes — slightly blurry, symmetrical, with no sharp edges. This is acceptable for gradient flow purposes but the blurriness directly determines the magnitude of the train/inference gap (the difference between R's output and the hard rasterizer's output). A blurry R means every stroke "leaks" slightly onto adjacent areas, which the agent learns to exploit (by placing strokes so their halos cover the target), and the hard rasterizer then renders as thin, misplaced strokes.

**Prevention:**
- Add a perceptual loss (e.g., VGG feature matching or simple gradient-domain loss) to R's training in addition to MSE. This sharpens edges and reduces the train/inference gap.
- Monitor the "blurriness metric" during R's training: compute the variance of the pixel gradient magnitude of R's output strokes. Higher variance = sharper edges = better.
- Keep the hard rasterizer's stroke borders genuinely sharp (no anti-aliasing) to give R a sharp training target.

**Warning signs:** The timelapse looks "watercolor-like" even for strokes that should be opaque rectangles; the final hard-rasterizer replay looks noticeably crisper than any frame of the training loop.

**Phase:** Renderer pre-training.

---

### S-2: Opaque Composition and the Non-Differentiable Ordering Problem

**What goes wrong:** Within a bundle of k=5 strokes, the composition order determines occlusion. R outputs each stroke independently. The composition step (painting stroke 1 onto canvas, then stroke 2 on top, etc.) is done with hard alpha-compositing: `canvas = stroke * mask + canvas * (1 - mask)`. This operation IS differentiable with respect to stroke pixel values, but the gradient through the ordering itself is zero — stroke 2 completely overwrites stroke 1 in the overlap region, and there is no gradient signal telling stroke 1 "you should be painted after stroke 2."

The agent must implicitly learn ordering through trial and error. For bundles with heavy overlap this is very slow.

**Prevention:**
- Accepted behavior — the project design document explicitly notes this is "non-resolved, known and accepted."
- However, it implies that the agent needs more training steps than if ordering were differentiable. Budget accordingly.
- As a practical mitigation: initialize R's opacity output close to 1 (fully opaque) to keep the composition clean and avoid half-transparent "ghost" strokes that create ambiguous ordering gradients.

**Warning signs:** Agent produces bundles where strokes are always spatially separated (it learned to avoid the ordering problem by not overlapping); timelapse has obvious "spatial partitioning" artifacts.

**Phase:** RL training, ongoing. Relevant at bundle design stage.

---

### S-3: Train/Inference Color Shift From Palette Projection

**What goes wrong:** The actor optimizes continuous RGB. At inference, the nearest-neighbor palette projection snaps each color to one of ~40 discrete values. If the learned policy consistently produces colors near palette boundaries (e.g., RGB = [0.45, 0.32, 0.21] equidistant between two palette entries), palette projection is effectively random at that point — the projected color can vary significantly between similar inputs.

For a 40-color palette, each color "cell" in RGB space has radius approximately `1/(40^(1/3)) ≈ 0.3` in each channel. For images with fine color gradients (skin tones, sky gradients), this rounding error is visible.

**Prevention:**
- During agent evaluation only (not training), compute the mean L2 distance between the actor's continuous RGB output and the projected palette color, across a test episode. This is the "palette quantization error." If > 0.05 per channel, the palette is too coarse or the agent is poorly calibrated.
- To reduce the gap: ensure the palette covers the training image distribution. If training on natural images, use a palette derived from k-means on the dataset, not a hand-picked color set.
- The Gumbel-softmax upgrade (in the evolution plan) eliminates this entirely — only needed if quantization error is visually noticeable.

**Warning signs:** Final eval images look "posterized"; specific hues (neutrals, skin tones) are consistently wrong by 10–20% intensity.

**Phase:** Evaluation pipeline, visual QA.

---

### S-4: Batch Normalization Train/Eval Mode in Critic

**What goes wrong:** If the critic or actor uses BatchNorm, the distinction between `model.train()` and `model.eval()` matters critically in RL. During training, BatchNorm uses batch statistics; during policy inference (action selection), it should use running statistics (`eval()` mode). The bug: when using the same network for both "select action" and "update critic," calling `model.train()` before the update and forgetting to switch back to `eval()` for the next action selection causes the batch statistics to be computed over a single sample (the current state) rather than a batch — leading to wildly incorrect normalization.

**Prevention:**
- Prefer LayerNorm over BatchNorm in the actor and critic for RL tasks. LayerNorm does not have train/eval mode distinction.
- If BatchNorm is used, wrap action selection with `with torch.no_grad(): model.eval(); action = actor(state); model.train()`.
- Log `actor.training` at the start of each action selection call during debug runs.

**Warning signs:** Action selection produces high-variance outputs on the same input; stochastic policy behavior that shouldn't be stochastic.

**Phase:** Actor/critic implementation.

---

### S-5: detach() Missing on Target Network Computation

**What goes wrong:** When computing the TD target `y = r + γ * Q_target(s', π_target(s'))`, the target network outputs must be detached from the computational graph before computing the critic loss. If `.detach()` is omitted on either `Q_target` or `π_target(s')`, PyTorch backpropagates through the target networks during the critic update, which:
1. Modifies target network weights (defeating their purpose as stable targets)
2. Creates a gradient loop that dramatically increases memory usage and may cause NaN

**Prevention:**
- Compute targets inside `with torch.no_grad():` — this is cleaner than manual `.detach()` calls because it prevents forgetting one of multiple paths.
- Add an assertion: `assert not target_actor.training` (target networks should always be in eval mode).
- This is a well-known DDPG implementation bug found even in published reference implementations (PyTorch forums confirm this is the single most common DDPG mistake).

**Warning signs:** GPU memory grows continuously over training steps; critic loss oscillates without converging; target network parameters change unexpectedly (compare parameter norms at step 0 and step 1000 — they should change only via soft update, not via gradient).

**Phase:** DDPG training loop implementation.

---

### S-6: Image Loading Without Consistent Normalization Across Pipeline Stages

**What goes wrong:** OpenCV loads images as BGR uint8; PIL loads as RGB uint8; torchvision transforms assume RGB float32 in [0,1]. If different modules use different loaders without explicit normalization contracts:
- Hard rasterizer draws in [0,1] float
- Target image loaded as [0,255] uint8
- R was trained on [0,1] float inputs
- Reward computed as L2 between mixed-scale tensors

The reward becomes meaningless and the actor's color outputs map to the wrong range.

**Prevention:**
- In `config.py`, define `IMAGE_RANGE = (0.0, 1.0)` and `IMAGE_DTYPE = torch.float32` as hard constants.
- All data loading functions return normalized float32 tensors in [0,1], always. Assert on shape (3, H, W) and range at load time.
- In the hard rasterizer, input params `(r,g,b)` are in [0,1], and output canvas is float32 [0,1].
- Run the full pipeline on a single test image before any training: assert `canvas.min() >= 0`, `canvas.max() <= 1.0`, same for target.

**Warning signs:** Reward values in the hundreds (pixel-space L2); actor outputs `r=g=b=0.5` for all strokes (regression to mean of wrongly-scaled range); hard rasterizer produces white-only or black-only canvases.

**Phase:** Hard rasterizer implementation (day 1), integration testing before RL.

---

## Phase-Specific Warnings

| Phase | Topic | Likely Pitfall | Mitigation |
|-------|-------|---------------|------------|
| Hard rasterizer | Parameterization convention | CM-7: Coordinate space mismatch between rasterizer and R | Define canonical [0,1] params in config.py on day 1 |
| Hard rasterizer | Data types | S-6: Mixed normalization | Assert float32 [0,1] output in rasterizer unit test |
| Renderer pre-training | Architecture | CM-5: Checkerboard artifacts from PixelShuffle | Visual inspection of 20+ strokes before declaring R ready |
| Renderer pre-training | Training coverage | CP-3: OOD strokes at RL time | Add extreme-parameter biased sampling (20% of batches) |
| Renderer pre-training | Sharpness | S-1: Blurry R inflates train/inference gap | Add gradient-domain loss term to R's training |
| Renderer pre-training | Freeze verification | CP-1: Accidental gradient flow through R | `assert not any(p.requires_grad for p in R.parameters())` immediately after freezing |
| Environment / `env.py` | Bundle composition | CP-5: Parallel vs sequential stroke application | Loop, don't batch; assert canvas changes after each stroke |
| Environment / `env.py` | State representation | CM-6: Missing step counter | Include normalized step fraction as input channel |
| Environment / `env.py` | Reward scale | CM-1: Reward in pixel units | Verify `abs(reward).mean() < 0.1` after first 100 steps |
| Actor / `actor.py` | Action normalization | CM-2: Inconsistent action ranges | All outputs in [0,1] via sigmoid; scale inside rasterizer |
| Critic / `critic.py` | Architecture | CP-4: Standard Q(s,a) instead of model-based V(s') | Critic takes rendered next-state image, not raw action params |
| DDPG training loop | Gradient bookkeeping | S-5: Missing detach on target outputs | Wrap all target network calls in `torch.no_grad()` |
| DDPG training loop | Gradient flow | CP-1: R unfreezing during actor update | Freeze verification assertion at top of each training epoch |
| DDPG training loop | Normalization layer | S-4: BatchNorm mode confusion | Use LayerNorm instead of BatchNorm in actor/critic |
| DDPG training loop | Q stability | CP-2: Q-value explosion | Log Q_max every 100 steps; clip critic gradients at 1.0 |
| DDPG training loop | Exploration | CM-3: Noise too large/too small | Log action std per episode; schedule σ decay |
| Replay buffer | Memory | CM-4: OOM on large buffer | Store as uint8, convert at sample time; start with 200k capacity |
| Evaluation (`eval.py`) | Color accuracy | S-3: Palette quantization error | Measure mean L2 between continuous RGB and projected palette color |
| Evaluation (`eval.py`) | Visual gap | CP-3 / S-1: Train vs eval render mismatch | Side-by-side comparison of R canvas vs hard rasterizer replay |

---

## Sources

- Huang et al. (2019), "Learning to Paint With Model-Based Deep Reinforcement Learning" — https://arxiv.org/pdf/1903.04411
- ar5iv rendered version of the paper (technical details) — https://ar5iv.labs.arxiv.org/html/1903.04411
- Original repository — https://github.com/hzwer/ICCV2019-LearningToPaint
- GitHub Issue #42: Renderer training doubts — https://github.com/hzwer/ICCV2019-LearningToPaint/issues/42
- GitHub Issue #23: RAM OOM on training — https://github.com/megvii-research/ICCV2019-LearningToPaint/issues/23
- Matheron (2020), "The Problem With DDPG" — https://arxiv.org/pdf/1911.11679
- OpenAI Spinning Up DDPG — https://spinningup.openai.com/en/latest/algorithms/ddpg.html
- PyTorch Forums: Gradient in DDPG actor backward — https://discuss.pytorch.org/t/question-about-gradient-calculation-in-backward-of-actor-network-of-ddpg/215917
- Checkerboard-free sub-pixel convolution — https://arxiv.org/pdf/1707.02937
- Distill: Deconvolution and Checkerboard Artifacts — https://distill.pub/2016/deconv-checkerboard/
- Investigation of BN in off-policy actor-critic — https://arxiv.org/html/2509.23750
