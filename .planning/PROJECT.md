# Paint AI — Mirabilia Épisode 1

## What This Is

Réimplémentation du papier "Learning to Paint" (DDPG + renderer neuronal différentiable).
Un agent RL apprend à peindre une image cible en posant des traits rectangulaires opaques sur une toile 64×64, guidé par un signal L2 incrémental.
Projet documenté publiquement dans la série de contenu **Mirabilia** — chaque épisode trace la progression de la baseline jusqu'aux évolutions.

## Core Value

L'agent produit un timelapse demo-able : on peut filmer l'IA qui peint une image cible trait par trait, de façon reconnaissable.

## Requirements

### Validated

(Aucun encore — à valider à l'envoi)

### Active

- [ ] Renderer dur fonctionnel : `draw(canvas, params)` → rectangle opaque orienté sur canvas
- [ ] Renderer neuronal R pré-entraîné : `params → image du trait seul`, repro visuelle validée
- [ ] Environnement RL : `reset()` / `step(action) → (state, reward, done)`, canvas 64×64 RGB
- [ ] DDPG complet : actor, critic, target networks, replay buffer, boucle d'entraînement
- [ ] Récompense L2 incrémentale : `r_t = L2(canvas_{t-1}, cible) − L2(canvas_t, cible)`
- [ ] Évaluation : projection palette nearest-neighbor + rendu final via rasterizer dur
- [ ] Timelapse générable : export frame-by-frame de l'agent peignant une image cible

### Out of Scope (épisode 1)

- WGAN / reward adversariale — évolution prévue si "bouillie floue" symptôme détecté
- Gumbel-softmax sur palette — évolution si projection palette dégrade visiblement
- Résolution > 64×64 — passage à 128×128 dans un épisode futur
- Coarse-to-fine multi-échelle — au-delà de 128×128, pas pour la baseline
- TD3 / SAC — remplacement DDPG seulement si instabilité critique confirmée
- Stop signal appris — seuil d'arrêt a posteriori (gain L2/trait < ε) uniquement

## Context

- Papier de référence : "Learning to Paint with Model-Based Deep Reinforcement Learning" (Zhewei Huang et al.) — PDF présent dans le dossier
- Simplifications délibérées vs le papier : traits rectangulaires opaques (vs arcs transparents), palette discrète ~40 couleurs, résolution réduite 64×64
- Deux renderers coexistants par conception : R neuronal (flou, différentiable, pour le gradient RL) + rasterizer dur (net, opaque, pour la vérité terrain et le rendu final)
- Composition de N traits opaques hors réseau (empilement) — gradient d'ordre de composition non résolu, connu et accepté
- L'agent décide k=5 traits simultanément depuis le même état (sans voir le canvas mis à jour intra-bundle)

## Constraints

- **Tech stack**: PyTorch + CUDA (GPU NVIDIA local)
- **Image size**: 64×64 — itération rapide, montée en résolution dans épisodes futurs
- **Stroke params**: `(cx, cy, w, h, θ, r, g, b)` — STROKE_DIM=8
- **Bundle size**: k=5 traits/step — décision verrouillée baseline
- **Reward**: L2 incrémental uniquement — pas de WGAN avant que la baseline fonctionne
- **Palette**: ~40 couleurs, projection nearest-neighbor RGB — décision verrouillée baseline
- **N_STROKES**: fixe et élevé — seuil d'arrêt a posteriori, pas de signal stop appris

## Key Decisions

| Décision | Rationale | Outcome |
|----------|-----------|---------|
| Traits rectangulaires opaques (vs arcs transparents) | Simplification, implémentation plus simple | — Pending |
| Renderer R neuronal différentiable (figé après pré-entraînement) | Rend le rendu différentiable → gradient exploitable par DDPG | — Pending |
| k=5 traits par step (bundle) | Raccourcit l'épisode → crédit assigné plus facilement, moins de variance | — Pending |
| L2 incrémental comme récompense | Signal dense à chaque bundle, simple à débugger | — Pending |
| Projection palette nearest-neighbor (pas Gumbel-softmax) | DDPG = actions continues ; Gumbel-softmax inadapté | — Pending |
| Rasterizer dur conservé | Vérité terrain pour pré-entraîner R + rendu final net | — Pending |
| 64×64 pour démarrer | Itération rapide — montée en résolution = évolution explicite | — Pending |
| Palette saisie manuellement depuis un mixeur physique | Les couleurs reflètent une vraie palette de peinture physique ; pas de palette algorithmique prédéfinie | — Pending |
| Colorspace de projection configurable (RGB / okLab / HSV) | okLab et HSV évitent les aberrations perceptuelles de la distance Euclidienne RGB ; choix configurable dans config.py | — Pending |

## Evolution

Ce document évolue aux transitions de phase et aux jalons de milestone.

**Après chaque transition de phase** (via `/gsd-transition`) :
1. Requirements invalidés ? → Déplacer en Out of Scope avec raison
2. Requirements validés ? → Déplacer en Validated avec référence de phase
3. Nouveaux requirements émergés ? → Ajouter en Active
4. Décisions à logger ? → Ajouter en Key Decisions
5. "What This Is" toujours exact ? → Mettre à jour si dérive

**Après chaque milestone** (via `/gsd-complete-milestone`) :
1. Review complète de toutes les sections
2. Core Value check — toujours la bonne priorité ?
3. Audit Out of Scope — raisons toujours valides ?
4. Mettre à jour Context avec état courant

---
*Last updated: 2026-06-08 after initialization*
