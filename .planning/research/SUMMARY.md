# Research Summary — Paint AI (Episode 1)

**Synthesized:** 2026-06-08
**Sources:** STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md, PROJECT.md

---

## Executive Summary

Paint AI est une réimplémentation de Huang et al. (ICCV 2019) "Learning to Paint" : un agent DDPG qui reproduit des images cibles en posant séquentiellement des traits rectangulaires opaques sur une toile 64×64, guidé par une récompense L2 incrémentale dense. Le composant central est un design à deux renderers : un rasterizer dur (déterministe, vérité terrain, sans gradient) et un renderer neuronal R figé (différentiable, entraîné une fois en supervisé). R est le pivot du pipeline — il permet aux gradients de la loss de l'acteur de remonter à travers l'étape de rendu. Sans un R bien entraîné, la boucle RL ne produit aucun signal utile.

L'approche recommandée est de tout construire from scratch en PyTorch — aucune librairie RL (pas SB3, pas TorchRL). L'environnement custom est trop non-standard pour que les wrappers aident, et le critic model-based (qui prend l'image du next-state rendu plutôt que les params d'action bruts) n'est supporté par aucune implémentation DDPG off-the-shelf. Le build est strictement séquentiel : rasterizer dur d'abord, puis pré-entraînement de R avec validation visuelle comme hard gate, puis DDPG. Sauter le gate est la décision la plus risquée qu'un développeur puisse prendre.

---

## Stack

- **Python 3.11 + PyTorch 2.7.0 (bundle CUDA 12.6)** — `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`. Pas de CUDA toolkit séparé nécessaire sur Windows.
- **DDPG from scratch en PyTorch** — ~200 lignes. Ne pas utiliser Stable-Baselines3 ou TorchRL.
- **Neural Renderer R : FC + decoder ConvTranspose2d** — Input : 8-dim stroke params. Output : 3×64×64 stroke image. Entraîné en supervisé MSE contre le rasterizer dur, puis figé. Sans BatchNorm.
- **TensorBoard pour monitoring, imageio pour export timelapse** — Local, zéro config. `imageio-ffmpeg` pour MP4.
- **Conda env, scripts `.py` plain pour l'entraînement** — Jupyter uniquement pour le prototypage du renderer.

---

## Table Stakes (must-haves v1)

Toutes requises. Aucune déférable. La chaîne de dépendances est stricte.

| Feature | Notes |
|---------|-------|
| Hard rasterizer `draw(canvas, params)` | Vérité terrain. Rectangle opaque orienté. STROKE_DIM=8 : (cx,cy,w,h,θ,r,g,b). Sans gradient. |
| Neural renderer R (pré-entraîné, figé) | Proxy différentiable. Supervisé sur paires (params → image du trait). Figé avant RL. |
| Générateur de dataset de pré-entraînement | Sampling aléatoire uniforme sur [0,1]^8. Pas de données humaines. Généré à la volée. |
| Environnement RL `env.py` | État : concat(canvas, target, step_channel) = 7×64×64. Applique k=5 traits séquentiellement via R. |
| Récompense L2 incrémentale | `r_t = (L2_prev - L2_new) / L2_init`. Normalisée. Dense par bundle. |
| Actor (CNN → action continue 40-dim) | Input 7×64×64 → 40 stroke params in [0,1]. Tous les k=5 traits depuis le même snapshot d'état. |
| Critic model-based (prend le next-state rendu) | Input : image du next-state rendu (6×64×64), PAS les params d'action bruts. Variante model-based du papier. |
| Target networks (actor + critic) | Soft update τ=0.005. Init deepcopy. Toujours en eval mode. |
| Replay buffer | Buffer circulaire. Stocker canvas en uint8, convertir float32 au sample. Commencer à 200k capacité. |
| Bruit d'exploration | Gaussien σ annealed 0.3 → 0.05. Logger l'écart type d'action par épisode. |
| Composition bundle (k=5 séquentiel) | Boucle, PAS batché en parallèle. Canvas mis à jour après chaque trait dans le bundle. |
| Palette discrète + nearest-neighbour | ~40 couleurs. Projection à l'eval uniquement. L'agent s'entraîne sur RGB continu. |
| Boucle d'entraînement `train.py` | 96 envs parallèles (tensor ops batchées). Logger Q_max, reward, critic loss toutes les 100 steps. |
| Pipeline eval `eval.py` | Replay via rasterizer dur pour output final. Projection palette. Frames timelapse. |
| Export timelapse | PNG frame-by-frame → GIF/MP4 via imageio. Livrable central de Mirabilia. |
| Module config `config.py` | `IMAGE_RANGE=(0.0,1.0)`, `STROKE_DIM=8`, `STROKES_PER_STEP=5`, `IMG_SIZE=64`. |

---

## Watch Out For

**CP-1 : Gradient qui fuit à travers R figé** — Bug silencieux le plus dangereux. R doit être figé avec à la fois `requires_grad_(False)` ET `torch.no_grad()` aux call sites. Le symptôme est indiscernable de "DDPG ne converge pas". Ajouter une assertion vérifiant que la norme des paramètres de R n'a pas changé depuis son checkpoint post-pré-entraînement.

**CP-2 : Explosion de la Q-value du critic DDPG** — Le target TD bootstrappé crée un feedback positif dans l'espace d'action 40-dim. Clipper les gradients du critic (`max_norm=1.0`). Logger `Q_max` toutes les 100 steps — doit rester sous ~1.0 pour une récompense L2 normalisée. Pas de récupération une fois que les target networks sont corrompus ; nécessite un restart complet. Échappatoire : TD3.

**CP-3 : R exploité sur des stroke params OOD** — L'agent RL découvrira des combinaisons de paramètres que R n'a jamais vus. R sort du garbage là-dessus ; l'agent reçoit un gradient comme si c'était valide. Prévention : biaiser 20% des batches d'entraînement de R vers des valeurs extrêmes (traits fins, traits plein-canvas, rotations extrêmes). Faire un audit de coverage avant RL.

**CP-4 : Critic Q(s,a) standard au lieu de V(s') model-based** — Presque tous les tutoriels implémentent `Q(state, action)`. Le critic du papier prend le **next-state rendu** en input. Mauvaise architecture : l'entraînement semble progresser mais la qualité de peinture plafonne loin sous les résultats du papier. Vérifier : le critic reçoit-il un tenseur image 6×64×64, pas des floats d'action ?

**CP-5 : Application parallèle des traits dans le bundle** — Tous les k=5 traits doivent être appliqués en boucle séquentielle. Appliquer tous les traits contre le même canvas de base simultanément ignore l'occlusion. Symptôme : l'agent concentre toutes les informations de couleur dans le dernier trait de chaque bundle.

---

## Architecture Overview

Deux pipelines séquentiels ; la Phase 1 est un hard gate avant la Phase 2.

**Phase 1 — Pré-entraînement du Renderer (une fois, bloquant)**
```
Params aléatoires [0,1]^8
    → Hard Rasterizer → image du trait ground-truth
    → Neural Renderer R → image du trait prédite
    → MSE loss → backprop dans R → répéter jusqu'à convergence
    → figer R, sauvegarder renderer.pkl
```
Validation visuelle requise : vérifier les traits fins, inclinés, en bord de frame.

**Phase 2 — Entraînement DDPG RL**
```
Image cible → env.reset() → obs (7×64×64)
    → Actor(obs) → action (40 floats) + bruit
    → env.step() : boucle k=5, R(stroke_i), composite sur canvas
    → reward = ΔL2 / L2_init
    → replay buffer.push(obs, action, reward, next_obs, done)
    → agent.update_policy() : critic MSE + policy gradient de l'acteur à travers R + soft-update targets
```

**Phase 3 — Inférence**
```
Actor entraîné → rollout déterministe → collecter stroke params
    → projection nearest-neighbour palette
    → replay via rasterizer dur → export timelapse
```

**Ordre de build (dépendances strictes par tier) :**

| Tier | Composants |
|------|-----------|
| 1 (sans dépendances) | `config.py`, `palette.py`, hard rasterizer `renderer.py` |
| 2 (nécessite T1) | `models/renderer.py`, `reward.py`, `ddpg/replay_buffer.py` |
| 3 (nécessite T1+2) | `pretrain_renderer.py`, `models/actor.py`, `models/critic.py` |
| **HARD GATE** | Valider R visuellement sur params OOD avant de continuer |
| 4 (nécessite T3) | `env.py`, `ddpg/agent.py` |
| 5 (nécessite T4) | `train.py`, `eval.py` + timelapse |

---

## Key Tensions

**Critic model-based vs. tutoriels DDPG standard** — Presque toutes les références implémentent `Q(s,a)`. Le papier utilise le next-state rendu. Facile à rater, difficile à diagnostiquer quand c'est faux. Doit être un point de décision explicite dans la Phase 3.

**Qualité de R vs. vitesse d'entraînement** — R MSE-only s'entraîne plus vite mais produit des traits flous, élargissant le gap train/inférence. Une loss perceptuelle en domaine gradient affûte R mais ajoute de la complexité au pré-entraînement. MSE-only est acceptable pour l'Episode 1 mais doit être monitoré.

**1 env pour le debug vs. 96 envs pour la vitesse** — Commencer avec 1 env, valider la boucle complète end-to-end, puis vectoriser. L'implémentation de référence tourne 96 envs parallèles. Ne pas vectoriser avant que la boucle soit validée.

---

## Roadmap Signals

Structure suggérée en 5 phases suivant les tiers de build, avec des hard gates de validation explicites.

| Phase | Nom | Livrables | Pitfalls clés |
|-------|-----|-----------|---------------|
| 1 | Foundation | `config.py`, `palette.py`, rasterizer dur. Test d'intégration normalisation. | S-6 (normalisation mixte), CM-7 (coord space) |
| 2 | Neural Renderer | Architecture R, pré-entraînement, audit coverage OOD, validation visuelle (hard gate). | CP-3 (coverage OOD), CM-5 (checkerboard), CP-1 (freeze verify) |
| 3 | DDPG Models | Actor, critic model-based, target networks, replay buffer (shape validation uniquement, pas d'entraînement). | CP-4 (mauvaise archi critic), S-4 (LayerNorm pas BatchNorm), CM-2 (action normalization) |
| 4 | Training Loop | `env.py`, agent, `train.py` ; 1 env d'abord puis 96 ; augmentation de reward confirmée. | CP-1, CP-2, CP-5, CM-1, CM-3, CM-4, CM-6, S-5 |
| 5 | Eval + Timelapse | `eval.py`, projection palette, replay rasterizer dur, GIF/MP4. Livrable : timelapse reconnaissable. | S-3 (quantization palette), gap train/eval visuel |

**Gaps à adresser lors du planning :**
- Dataset d'images d'entraînement non encore spécifié — n'importe quel dataset d'images fonctionne à 64×64 (resize + normalize). À fixer en Phase 4.
- Stratégie de checkpoints non définie — fréquence, quel checkpoint pour l'eval. À fixer en Phase 4/5.
- Critère d'arrêt du pré-entraînement de R qualitatif — définir un seuil concret (ex : val MSE < 0.005) en Phase 2.
