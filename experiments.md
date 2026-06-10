# Autoresearch — Renderer Experiments

Chaque ligne = un cycle autoresearch. L'agent ajoute une ligne après chaque expérience.
`best` = meilleur `final_val_mse` observé jusqu'ici (200 steps, mode `--quick`).

| # | Change | Val MSE (200 steps) | vs best | Kept |
|---|--------|---------------------|---------|------|
| 0 | baseline — fc=512, ch=128→64→32→16, ReLU, bilinear, lr=1e-3, bs=1024, EXTREME_FRAC=0.2 | 0.053452 | — | baseline |
| 1 | lr 1e-3 → 3e-3 | 0.057252 | +0.003800 ❌ | revert |
| 2 | lr 1e-3 → 2e-3 | 0.056552 | +0.003100 ❌ | revert |
| 3 | lr 1e-3 → 5e-4 | 0.057337 | +0.003885 ❌ | revert |
| 4 | scheduler ReduceLROnPlateau → CosineAnnealingLR(T_max=200) | 0.056395 | +0.002943 ❌ | revert |
| 5 | EXTREME_FRAC 0.2 → 0.3 | 0.055710 | +0.002258 ❌ | revert |
| 6 | EXTREME_FRAC 0.2 → 0.4 | 0.057492 | +0.004040 ❌ | revert |
| 7 | EXTREME_FRAC 0.2 → 0.15 | 0.056837 | +0.003385 ❌ | revert |
| 8 | BATCH_SIZE 1024 → 2048 | 0.056038 | +0.002586 ❌ | revert |
| 9 | BATCH_SIZE 1024 → 512 | 0.061805 | +0.008353 ❌ | revert |
| 10 | channels 256→128→64→32 (fc=1024) | 0.056219 | +0.002767 ❌ | revert |
| 11 | channels 64→32→16→8 (fc=256) | 0.055625 | +0.002173 ❌ | revert |
| 12 | FC 8→1024→512 (two linear layers) | 0.058544 | +0.005092 ❌ | revert |
| 13 | FC=256 → view(-1,64,2,2), stage1=Conv(64→64) | 0.057962 | +0.004510 ❌ | revert |
| 14 | Activation ReLU → GELU | 0.056851 | +0.003399 ❌ | revert |
| 15 | Activation ReLU → LeakyReLU(0.2) | 0.059786 | +0.006334 ❌ | revert |
| 16 | Activation ReLU → ELU | 0.057988 | +0.004536 ❌ | revert |
| 17 | Upsampling nearest stages 1-3 (bilinear stage4) | 0.058074 | +0.004622 ❌ | revert |

## Série 2 — fg-weighted MSE loss (FG_WEIGHT=50, `1 + 49 * mask`)

> ⚠️ Expériences 0–17 invalides : loss plain MSE → modèle tout-noir (prédictions max≈0.07).
> Fix appliqué le 2026-06-10 : `weight = 1 + 49 * (targets > 0.01)` dans la training loop.
> Nouvelle baseline mesurée avec la fg-weighted loss. Val MSE = MSE **plain** sur le val set (comparable entre exps).
> Objectif : bords plus rectangulaires + meilleure reconstruction couleur.

| # | Change | Val MSE (200 steps) | vs best | Kept |
|---|--------|---------------------|---------|------|
| 18 | **NOUVELLE BASELINE** — même archi, loss fg-weighted (weight=50) | 0.062339 | — | baseline |
| 19 | FG weight 50→100 | 0.143007 | +0.080668 ❌ | revert |
| 20 | FG weight 50→20 | 0.038102 | −0.024237 ✅ | kept → new best |
| 21 | double conv par stage (2 convs après chaque upsample) | 0.039274 | +0.001172 ❌ | revert |
| 22 | channels 256→128→64→32, FC=1024 (view 256×2×2) | 0.040552 | +0.002450 ❌ | revert |
| 23 | stage5 refinement 64×64 (conv supplémentaire) | 0.039743 | +0.001641 ❌ | revert |
| 24 | FC=1024, view(-1,256,2,2), ch 256→64→32→16→16 | 0.035434 | −0.002668 ✅ | kept → new best |
| 25 | FC=1024 + double convs | 0.041980 | +0.006546 ❌ | revert |
| 26 | FG weight 20→10 (avec FC=1024) | 0.022707 | −0.012727 ✅ | **BEST** kept |
| 27 | FG weight 10→5 | 0.022892 | +0.000185 ❌ | revert (dans le bruit) |
| 28 | LR 1e-3→5e-4 | 0.029635 | +0.006928 ❌ | revert |
| 29 | LR 1e-3→2e-3 | 0.027634 | +0.004927 ❌ | revert |
| 30 | EXTREME_FRAC 0.2→0.1 | 0.031294 | +0.008587 ❌ | revert |
| 31 | stage4 nearest-neighbor (vs bilinear) | 0.028159 | +0.005452 ❌ | revert |
| 32 | 5 stages : ×4 → deux ×2 (stages 4+5) | 0.029419 | +0.006712 ❌ | revert |
| 33 | loss MSE→L1 fg-weighted | 0.036222 | +0.013515 ❌ | revert |
| 34 | BATCH_SIZE 1024→2048 | 0.030495 | +0.007788 ❌ | revert |
| 35 | stage1 : 256→128 (vs 256→64) | 0.032278 | +0.009571 ❌ | revert |
| 36 | pas de scheduler (LR constant) | 0.028684 | +0.005977 ❌ | revert |
| 37 | ReduceLROnPlateau patience=2 (vs 5) | 0.028203 | +0.005496 ❌ | revert |

**Best config (série 2)** : FC=1024→view(256,2,2), ch 256→64→32→16→16, FG weight=10, LR=1e-3, patience=5, BS=1024, EXTREME_FRAC=0.2 → **val MSE 0.022707** (200 steps, plain MSE)

## Série 3 — métrique fg-MSE + objectif qualité (hard alpha threshold en Phase 4)

> Changements 2026-06-10 :
> - Métrique val switchée vers **foreground MSE** (MSE sur pixels `target > 0.01` uniquement).
>   Mesure la précision couleur+position sur les pixels du trait — ignore le fond noir.
> - Contrainte taille levée : modèles plus grands autorisés.
> - Compositing Phase 4 : `alpha = (R_out.max > 0.3).float()` (hard threshold → bords francs).
>
> Baseline série 3 = best config série 2 re-mesuré avec fg-MSE.

| # | Change | Val fg-MSE (200 steps) | vs best | Kept |
|---|--------|------------------------|---------|------|
| 38 | **BASELINE série 3** — same best config, métrique fg-MSE | 0.010495 | — | baseline |
