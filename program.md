# Autoresearch — Renderer Pre-Training

## Objectif

Minimiser `final_val_mse` mesuré après **200 steps** (mode `--quick`).
Cible stretch : < 0.003 (vs 0.005 requis pour le full run).

Le mode `--quick` est le proxy : si une modification améliore le quick MSE, elle a de bonnes chances d'améliorer aussi le full run.

## Protocole loop (un cycle = une expérience)

1. Proposer **une seule modification** à la fois (architecture ou hyperparamètre — pas les deux)
2. Appliquer la modification dans le fichier concerné
3. Lancer : `python pretrain_renderer.py --mode quick`
4. Lire `latest_result.json` → noter `final_val_mse`
5. Comparer avec le meilleur MSE connu (`best_val_mse` ci-dessous)
6. **Si amélioration** : conserver, mettre à jour `best_val_mse`
7. **Si pas d'amélioration** : `git checkout -- models/renderer.py pretrain_renderer.py` (revert)
8. Ajouter une ligne dans `experiments.md`
9. Répéter

Tracker en mémoire pour la session :
- `best_val_mse` : meilleur `final_val_mse` observé (initialiser avec la baseline du step 0)
- `experiment_number` : incrémenter à chaque cycle

## Fichiers modifiables

- `models/renderer.py` — architecture NeuralRenderer (channels, activations, FC size, upsampling)
- `pretrain_renderer.py` — hyperparamètres d'entraînement (LR, EXTREME_FRAC, scheduler, BATCH_SIZE)

## Fichiers interdits (ne jamais toucher)

- `renderer.py` — hard rasterizer, ground truth, ne doit pas changer
- `config.py` — constantes partagées avec Phase 3/4
- `tests/` — tests de régression, ne pas modifier
- `experiments.md` — log seulement, ne pas réécrire l'historique

## Contraintes verrouillées

Ces règles s'appliquent à TOUTES les expériences, sans exception :

| Contrainte | Raison |
|---|---|
| `STROKE_DIM = 8` (input du renderer) | Locked baseline |
| `IMG_SIZE = 64` (output 64×64) | Locked baseline |
| Pas de `BatchNorm` | Incompatible avec single-sample inference pendant le RL (D-11) |
| Sortie finale `Sigmoid` → range [0,1] | Requis pour la compositing formula Phase 4 (D-01) |
| `weights_only=True` dans `torch.load` | Sécurité contre exécution de code arbitraire (T-02-PKL) |
| Stage 4 : `scale_factor=4` (pas 2) | Nécessaire pour atteindre 64×64 depuis 16×16 (Pitfall 1) |
| Génération des targets sur CPU | GPU est plus lent pour le rasterizer loop (Pitfall 3) |

## Axes d'exploration (par priorité suggérée)

### Architecture (models/renderer.py)
- Channels : `128→64→32→16` (baseline) → essayer `256→128→64→32` ou `64→32→16→8`
- FC size : `512` (baseline) → essayer `256`, `1024`
- Activation : `ReLU` → `GELU`, `LeakyReLU(0.2)`, `ELU`
- Upsampling mode : `bilinear` → `nearest` pour les premiers stages
- Ajouter une connexion résiduelle entre le FC et le stage final (si applicable)

### Entraînement (pretrain_renderer.py)
- LR : `1e-3` (baseline) → essayer `5e-4`, `2e-3`, `3e-3`
- Scheduler : `ReduceLROnPlateau(patience=5)` → `CosineAnnealingLR(T_max=QUICK_STEPS)`
- `EXTREME_FRAC` : `0.2` (baseline) → essayer `0.3`, `0.15`, `0.4`
- `BATCH_SIZE` : `1024` (baseline) → essayer `512`, `2048` (attention à la mémoire GPU)

## Lecture du résultat

```json
{
  "mode": "quick",
  "steps": 200,
  "final_val_mse": 0.00341,
  "baseline_val_mse": 0.08812,
  "elapsed_seconds": 187.4,
  "timestamp": "2026-06-10T23:14:02"
}
```

Seul `final_val_mse` compte pour la comparaison. `baseline_val_mse` = MSE du modèle non entraîné (varie selon l'architecture).

## Lancer le full run

Une fois la meilleure config trouvée :
```
python pretrain_renderer.py --mode full
```
Cela lance les 976 steps complets, sauvegarde `renderer.pkl`, et génère `visual_gate.png`.
