# Requirements — Paint AI (Mirabilia Épisode 1)

## v1 Requirements

### Foundation (FOUND)

- [x] **FOUND-01**: Le module `config.py` expose les constantes `IMG_SIZE=64`, `STROKE_DIM=8`, `STROKES_PER_STEP=5`, `N_STROKES` et `IMAGE_RANGE=(0.0, 1.0)` accessibles depuis tous les modules
- [x] **FOUND-02**: Le module `palette.py` définit la palette comme une simple liste RGB éditable manuellement (les couleurs sont saisies par l'utilisateur depuis un mixeur de peinture physique) et expose `project_color(rgb, colorspace) → palette_rgb` avec `colorspace ∈ {"rgb", "oklab", "hsv"}` — nearest-neighbor dans l'espace choisi ; `colorspace` configurable dans `config.py`
- [ ] **FOUND-03**: `renderer.py` (dur) expose `draw(canvas, stroke_params) → canvas` qui dessine un rectangle opaque orienté avec `params=(cx, cy, w, h, θ, r, g, b)` en pure tensor ops, sans autograd

### Neural Renderer (REND)

- [ ] **REND-01**: `models/renderer.py` implémente le réseau R avec architecture FC + decoder ConvTranspose2d : input `(batch, 8)` → output `(batch, 3, 64, 64)` image du trait seul (plage [0,1])
- [ ] **REND-02**: `pretrain_renderer.py` génère des paires (params aléatoires [0,1]^8 → image rasterizer dur), entraîne R par MSE loss, sauvegarde le checkpoint `renderer.pkl` ; 20% des batches biaisés vers des params extrêmes (traits fins, plein-canvas, rotations extrêmes)
- [ ] **REND-03**: Validation visuelle de R sur un set de traits de test (fins, inclinés, en bord de frame, extrêmes) avant tout entraînement RL ; une assertion vérifie que la norme des paramètres de R ne change pas après freeze

### DDPG Models (DDPG)

- [x] **DDPG-01**: `models/actor.py` implémente l'acteur CNN : input `(batch, 7, 64, 64)` (canvas+cible+step_channel) → output `(batch, 40)` en [0,1] via sigmoid, représentant les `k=5` traits
- [ ] **DDPG-02**: `models/critic.py` implémente le critic model-based V(s') : input l'image du next-state rendu `(batch, 6, 64, 64)` → output scalar Q ; PAS Q(s, a) standard (différence critique vs papier)
- [ ] **DDPG-03**: `ddpg/agent.py` crée les target networks (deepcopy de actor et critic), les maintient en `eval()` mode permanent, et implémente le soft update avec `τ=0.005`
- [ ] **DDPG-04**: `ddpg/replay_buffer.py` implémente un buffer circulaire de 200k transitions ; les canvas sont stockés en `uint8` et convertis en `float32` au sample pour économiser la RAM

### Training (TRAIN)

- [ ] **TRAIN-01**: `env.py` implémente `reset() → obs` et `step(action) → (obs, reward, done)` ; état = concat(canvas, cible, step_channel) en 7×64×64 ; applique les `k=5` traits en boucle séquentielle via R ; reward = `(L2_prev - L2_new) / L2_init` (normalisé)
- [ ] **TRAIN-02**: `ddpg/agent.py` implémente la boucle de mise à jour : critic loss MSE, policy gradient de l'acteur via R figé (`torch.no_grad()` sur R), soft update des target networks ; gradient clipping critic `max_norm=1.0`
- [ ] **TRAIN-03**: `train.py` lance l'entraînement sur 96 envs parallèles (tensor ops batchées), logge `Q_max`, reward moyen, et critic loss dans TensorBoard toutes les 100 steps ; exécutable en CLI avec des arguments configurables
- [ ] **TRAIN-04**: Le bruit d'exploration gaussien est annealed de `σ=0.3` à `σ=0.05` au cours de l'entraînement ; l'écart type moyen d'action est loggé par épisode

### Eval & Timelapse (EVAL)

- [ ] **EVAL-01**: `eval.py` exécute un rollout déterministe de l'acteur entraîné sur une image cible, projette les couleurs RGB continues sur la palette (nearest-neighbor), puis rejoue la liste ordonnée de stroke params via le rasterizer dur pour produire le canvas final net
- [ ] **EVAL-02**: `eval.py` exporte un timelapse frame-by-frame (PNG → GIF et/ou MP4 via imageio) montrant l'agent peindre l'image cible trait par trait ; livrable central de Mirabilia Épisode 1

---

## v2 Requirements (épisodes futurs — déclenché par symptôme)

| Feature | Symptôme déclencheur |
|---------|---------------------|
| WGAN / récompense adversariale | Agent produit une "bouillie floue", convergence vers L2 moyen |
| TD3 (double critic, target policy smoothing) | Instabilité DDPG, Q-value explosion non résolue par clipping |
| Gumbel-softmax sur palette | Projection palette dégrade visiblement le rendu final |
| Résolution 128×128 | 64×64 trop grossier pour les images cibles choisies |
| Coarse-to-fine multi-échelle | Artefacts de composition, agent ne gère pas bien le recouvrement |
| Stop signal appris | Trop de traits gaspillés sur zone déjà peinte |

---

## Out of Scope (Épisode 1)

- **WGAN / reward adversariale** — Complexité inutile avant une baseline L2 qui fonctionne
- **Gumbel-softmax palette** — DDPG = actions continues ; Gumbel-softmax inadapté à la baseline
- **Résolution > 64×64** — 64×64 = itération rapide ; montée en résolution = épisode dédié
- **Coarse-to-fine multi-échelle** — Hors scope ; quasi obligatoire seulement au-delà de 128×128
- **TD3 / SAC** — Remplacement DDPG uniquement si instabilité critique confirmée et non résolue
- **Stop signal appris** — Seuil d'arrêt a posteriori (gain L2/trait < ε) uniquement
- **Données d'images spécialisées** — Tout dataset images RGB standard convient à 64×64 (CIFAR, CelebA, etc.)

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FOUND-01 | Phase 1 — Foundation | Complete (Plan 01-01) |
| FOUND-02 | Phase 1 — Foundation | Complete (Plan 01-01) |
| FOUND-03 | Phase 1 — Foundation | Pending |
| REND-01 | Phase 2 — Neural Renderer | Pending |
| REND-02 | Phase 2 — Neural Renderer | Pending |
| REND-03 | Phase 2 — Neural Renderer | Pending |
| DDPG-01 | Phase 3 — DDPG Models | Complete (Plan 03-01) |
| DDPG-02 | Phase 3 — DDPG Models | Pending |
| DDPG-03 | Phase 3 — DDPG Models | Pending |
| DDPG-04 | Phase 3 — DDPG Models | Pending |
| TRAIN-01 | Phase 4 — Training Loop | Pending |
| TRAIN-02 | Phase 4 — Training Loop | Pending |
| TRAIN-03 | Phase 4 — Training Loop | Pending |
| TRAIN-04 | Phase 4 — Training Loop | Pending |
| EVAL-01 | Phase 5 — Eval & Timelapse | Pending |
| EVAL-02 | Phase 5 — Eval & Timelapse | Pending |
