# Paint AI — Choix de design et architecture

Réimplémentation simplifiée de *Learning to Paint* (DDPG + renderer neuronal).
Traits rectangulaires opaques, palette de ~40 couleurs.

## Choix de design

| Décision | Choix retenu | Raison |
|---|---|---|
| Forme des traits | Rectangles orientés, opaques | Simplification vs arcs transparents du papier |
| Renderer | **SoftRasterizer** analytique (sigmoid SDF, différentiable), **aucun entraînement requis** | Formule analytique : `alpha = sigmoid((w/2 - |dx'|)/β) * sigmoid((h/2 - |dy'|)/β)`. Différentiable, déterministe, pas de drift. Remplace le CNN neuronal initial après autoresearch (série 1–4). |
| Rasterizer dur | Conservé | Vérité terrain pour validation visuelle + rendu final net |
| Couleur | RGB continu en sortie de l'agent, projeté sur la palette (nearest-neighbor L2) en post-traitement | DDPG = actions continues ; Gumbel-softmax inadapté |
| Agent | DDPG model-free, espace d'action continu | Exigence ; rendu via `R` figé rapproche du model-based du papier |
| Nombre de traits | Fixe, élevé | Seuil d'arrêt a posteriori (gain L2/trait < ε) plutôt que signal stop appris |
| Traits par step | Bundle de **k = 5** | Raccourcit l'épisode d'un facteur k → crédit assigné plus facilement, moins de variance. Fidèle au papier |
| Récompense | **L2 incrémental** (baisse de L2 par step), négatif | Signal dense à chaque bundle. WGAN seulement en phase 2 |
| Image | 64×64 pour démarrer | Petit = itération rapide |

## Points de vigilance

- **SoftRasterizer vs Rasterizer dur :**
  - `SoftRasterizer` (différentiable) → pendant l'entraînement RL, pour le gradient. Bords contrôlés par `beta` (1.0 → ~4px de transition).
  - Rasterizer dur (net, opaque, ordre respecté) → rendu final, en rejouant la liste ordonnée de params.
- **Bords flous — voulu :**
  - `SoftRasterizer` produit des bords soft par construction (sigmoid SDF). C'est ce qui rend le gradient exploitable. Aucun mélange de couleur parasite : le masque alpha est calculé analytiquement, la couleur n'est jamais interpolée avec le fond — c'est `alpha * color + (1-alpha) * canvas`.
- **Écart train/inférence** : un trait "bon" avec `SoftRasterizer` aura des bords légèrement flous vs le rasterizer dur. Négligeable si les traits sont grands devant la bande de transition (4px à 64×64).
- **Occlusion / ordre de composition** : `R` rend *un* trait. La composition de N traits opaques (qui recouvre qui) est une étape séparée, gérée hors réseau par empilement. Le gradient à travers l'ordre de composition reste rugueux — non résolu par `R`.
- **Projection couleur** : l'agent optimise une couleur continue mais on lui impose la couleur de palette la plus proche à l'inférence → léger écart, faible si palette bien répartie.

## Structure des fichiers

```
paint_ai/
├── config.py              # hyperparams, dims action/état, N_STROKES, STROKE_DIM, taille image, STROKES_PER_STEP
├── palette.py             # les ~40 couleurs, projection nearest-neighbor RGB->palette
├── renderer.py            # rasterizer DUR : draw(canvas, params) -> rectangle opaque net. Vérité terrain + rendu final
├── env.py                 # environnement RL : reset, step(action)->(state, reward), canvas courant. État = concat(canvas, cible)
├── reward.py              # L2 négatif (incrémental). WGAN plus tard
├── models/
│   ├── renderer.py        # réseau R : params -> image du trait seul (+ masque). Différentiable, figé après pré-entraînement
│   ├── actor.py           # politique : état (canvas+cible empilés) -> action continue (params trait)
│   └── critic.py          # Q(état, action)
├── ddpg/
│   ├── agent.py           # DDPG : update actor/critic, target networks, soft update
│   └── replay_buffer.py   # mémoire de transitions
├── pretrain_renderer.py   # entraînement supervisé de R contre le rasterizer dur, puis figé
├── train.py               # boucle d'entraînement DDPG
├── eval.py                # peindre une cible, seuil d'arrêt, projection palette, rendu final via rasterizer dur
└── utils.py               # logging, sauvegarde, visualisation
```

### Contenu des pièces critiques

- **`renderer.py` (dur)** : `draw(canvas, stroke_params)` avec `params = (cx, cy, w, h, θ, r, g, b)`. Calcule les pixels du rectangle orienté, les écrase. Pure tensor ops, gradient non requis.
- **`models/renderer.py` (R)** : CNN/MLP, `params -> image du trait seul`. Sort le trait seul + masque (PAS le canvas composé) → composition gardée hors réseau, contrôle de l'occlusion conservé. Entraîné en supervisé contre le rasterizer dur sur des traits aléatoires, puis figé.
- **`env.py`** : état = `(canvas_courant, cible)` empilés en canaux (6×H×W si RGB). `step` applique un (ou un bundle de) trait(s) via `R`, recalcule le canvas, renvoie la baisse de L2.
- **`reward.py`** : récompense incrémentale `r_t = L2(canvas_{t-1}, cible) − L2(canvas_t, cible)`. PAS le L2 absolu. Calculée sur le canvas après le bundle entier.
- **`actor.py`** : CNN état (6×H×W) -> action ∈ [0,1]^(STROKE_DIM × STROKES_PER_STEP) = 40 dims avec k=5. Les k traits sont décidés simultanément depuis le même état (ils ne voient pas le canvas mis à jour par leurs voisins du même bundle).
- **`config.py`** : `N_STROKES`, `STROKE_DIM=8` (cx,cy,w,h,θ,r,g,b), `IMG_SIZE=64`, `STROKES_PER_STEP=5`.

## Ordre de travail

1. **`renderer.py` (dur)** — nécessaire comme vérité terrain de toute façon.
2. **`pretrain_renderer.py`** — entraîner `R`, vérifier visuellement la repro d'un trait isolé. Composant le moins risqué ; le valider tôt évite de débugger le RL avec un renderer cassé.
3. **DDPG par-dessus `R` figé** (`models/`, `ddpg/`, `env.py`, `reward.py`, `train.py`).
4. **`eval.py`** — projection palette + rendu final net.

## Évolutions possibles (par symptôme)

Chaque ligne = un problème probable, le diagnostic, et l'amélioration ciblée. Bon fil narratif pour montrer de la progression.

**SoftRasterizer**
- *Symptôme : bords trop flous, traits fins non visibles.* → Réduire `beta` (0.5 → ~2px de transition, plus net). Attention : trop petit = gradients plus abrupts.
- *Symptôme : gradients instables sur traits très fins.* → Augmenter légèrement `beta`, ou clipper les gradients à l'actor.
- *Symptôme : écart trop grand entre canvas RL et rendu final dur.* → Acceptable ; c'est structurel. Mitiger en re-seuillant l'alpha à l'inférence ou en passant le canvas dur comme état (au lieu du canvas soft).

**Agent / apprentissage**
- *Symptôme : l'agent ne converge pas, récompense plate.* → Vérifier l'échelle de la récompense (normaliser), réduire k, ajouter du bruit d'exploration (Ornstein-Uhlenbeck ou gaussien décroissant).
- *Symptôme : convergence vers une bouillie floue, traits moyennés.* → C'est le défaut connu de L2. Passer au **WGAN** (ajoute `models/discriminator.py`, récompense = score critique adversarial). C'est l'évolution majeure du papier.
- *Symptôme : crédit mal attribué, épisodes trop longs.* → Réduire `N_STROKES`, ou augmenter k, ou découper l'image en patches peints séparément (coarse-to-fine).
- *Symptôme : instabilité du critique (Q explose).* → Target networks + soft update τ plus petit, clipping de gradient, ou passer à **TD3** (double critique, target policy smoothing) — drop-in sur DDPG.

**Couleur / palette**
- *Symptôme : la projection palette dégrade visiblement le rendu.* → Passer en Gumbel-softmax sur les 40 couleurs (change l'espace d'action ; abandonne le RGB continu), ou apprendre la palette conjointement.

**Composition / occlusion**
- *Symptôme : l'ordre des traits produit des artefacts, l'agent ne gère pas le recouvrement.* → Passer le numéro d'étape / un canal "profondeur" dans l'état, ou peindre en coarse-to-fine (gros traits d'abord, fins ensuite).

**Échelle**
- *Symptôme : 64×64 trop grossier.* → Monter à 128×128 puis 256×256. Coarse-to-fine quasi obligatoire au-delà de 128 (diviser l'image en sous-régions, un agent par échelle comme dans le papier).

## Décisions verrouillées (ne pas rouvrir avant que la baseline marche)

- k = 5, L2 incrémental, pas de WGAN, projection palette, 64×64, nombre de traits fixe.
- Tout changement = une évolution de la liste ci-dessus, faite **une à la fois**, sur une baseline qui fonctionne, pour pouvoir comparer.
