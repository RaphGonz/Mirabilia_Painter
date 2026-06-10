# Autoresearch — Renderer Pre-Training

## Objectif

**Maximiser la qualité visuelle du renderer.** Pas de contrainte de vitesse d'inférence.

Métrique : `final_val_mse` = **foreground MSE** (MSE sur les pixels où `target > 0.01` uniquement).
Mesure la précision couleur + position sur les pixels du trait. Ignoré le fond noir.

Cible : < 0.05 sur le full run (976 steps). Stretch : < 0.03.

## Protocole loop (un cycle = une expérience)

1. Proposer **une seule modification** à la fois
2. Appliquer la modification dans le fichier concerné
3. Lancer : `python pretrain_renderer.py --mode quick`
4. Lire `latest_result.json` → noter `final_val_mse`
5. Comparer avec le meilleur MSE connu (`best_val_mse`)
6. **Si amélioration** : conserver, mettre à jour `best_val_mse`
7. **Si pas d'amélioration** : `git checkout -- models/renderer.py pretrain_renderer.py`
8. Ajouter une ligne dans `experiments.md`
9. Répéter

## Fichiers modifiables

- `models/renderer.py` — architecture NeuralRenderer (channels, activations, FC size, upsampling, profondeur)
- `pretrain_renderer.py` — hyperparamètres (LR, FG_WEIGHT, EXTREME_FRAC, BATCH_SIZE)

## Fichiers interdits

- `renderer.py` — hard rasterizer, ground truth
- `config.py` — constantes partagées Phase 3/4
- `tests/` — tests de régression
- `experiments.md` — log seulement

## Contraintes verrouillées

| Contrainte | Raison |
|---|---|
| `STROKE_DIM = 8` (input) | Locked baseline |
| `IMG_SIZE = 64` (output 64×64) | Locked baseline |
| Pas de `BatchNorm` | Incompatible single-sample inference RL (D-11) |
| Sortie finale `Sigmoid` → [0,1] | Requis compositing Phase 4 (D-01) |
| `weights_only=True` dans `torch.load` | Sécurité (T-02-PKL) |
| Génération targets sur CPU | GPU plus lent pour le rasterizer loop (Pitfall 3) |

> **Contraintes levées** : taille du modèle, nombre de paramètres, vitesse d'inférence.
> Objectif = renderer le plus précis possible, quelle que soit la taille.

## Best config actuel (baseline série 2)

- `fc: Linear(8, 1024) → view(-1, 256, 2, 2)`
- stages : Upsample bilinear + Conv 256→64→32→16→16
- FG weight = 10 (`weight = 1 + 9 * (targets > 0.01)`)
- LR = 1e-3, ReduceLROnPlateau(patience=5, factor=0.5)
- BATCH_SIZE = 1024, EXTREME_FRAC = 0.2

**val fg-MSE baseline** : à mesurer avec `--mode quick` (sera différent de la série 2 qui utilisait plain MSE).

## Axes prioritaires à explorer (nouvelle métrique fg-MSE)

### Architecture — taille

- FC plus grand : 2048, 4096
- Channels larges tout au long : 256→128→64→32 (au lieu de 256→64→32→16)
- Stage 0 supplémentaire : 1×1 → 2×2 avant stage 1 (plus de spatial info dès le départ)
- Stage 5 : refinement 64×64 conv supplémentaire

### Architecture — structure

- Connexions résiduelles entre stages (ResNet-style)
- Tête alpha séparée : sorties (RGB, mask), mask entraîné avec BCE vs `(target>0.01).float()`
- Convolutions 5×5 en stage 4 (grand receptive field à pleine résolution)

### Entraînement

- FG weight : tester 15, 12, 8 autour du best=10
- EXTREME_FRAC : 0.25, 0.3 (plus de cas difficiles)
- BATCH_SIZE : 512 (plus de gradient steps par epoch)
- Deux optimiseurs : Adam + weight decay (1e-4)

## Lancer le full run

```
python pretrain_renderer.py --mode full
```

976 steps, sauvegarde `renderer.pkl`, génère `visual_gate.png`.
