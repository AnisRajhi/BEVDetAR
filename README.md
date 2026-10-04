# lss_det v2 — détecteur 3D caméra LSS + CenterPoint

6 caméras → EfficientNet-B0 → Lift camera-aware (profondeur supervisée par lidar)
→ Splat (avec BEV augmentation) → BEV backbone → tête CenterPoint.

Tous les paramètres sont dans `lss_det/config.py` (en dur), avec une fonction
`validate()` qui vérifie leurs relations au chargement.

## Ordre d'exécution

```bash
# 0. Dépendances
pip install torch torchvision efficientnet_pytorch pyquaternion nuscenes-devkit pytest

# 1. Tests (~1 min CPU, aucune donnée nécessaire) — à relancer après toute modif
#    de transforms / geometry / depth_targets / bev_augmentation / dataset
python3 -m pytest -q tests

# 2. Audit des données (mesure ce que la v1 demandait d'impossible)
python3 audit_dataset.py

# 3. Sanity check : config.py -> OVERFIT_NUM_SAMPLES = 10
python3 train.py          # attendu : mAP élevé et dAcc proche de 1 sur ces 10 frames

# 4. Entraînement : config.py -> OVERFIT_NUM_SAMPLES = 0
python3 train.py

# 5. Évaluation d'un checkpoint (CHECKPOINT en tête de fichier)
python3 evaluate.py
```

Mémoire GPU insuffisante : dans `config.py`, `IMAGE_SIZE = (128, 352)`.
Les anciens checkpoints ne sont pas compatibles (architecture et grille BEV différentes).

## Entraîner sur une partie de nuScenes trainval

mini ne contient que 8 scènes d'entraînement et 2 de validation : la profondeur cesse de
généraliser dès l'epoch 8 environ. Pour plus de diversité :

1. Télécharger les **métadonnées v1.0-trainval** et **une ou plusieurs archives de blobs**
   trainval (environ 85 scènes chacune ; des archives « keyframes » plus légères peuvent
   exister, voir la page de téléchargement nuScenes). Les extraire dans `DATAROOT`.
2. Dans `config.py` : `VERSION = "v1.0-trainval"`.
3. `python3 audit_dataset.py`, puis `python3 train.py`.

Le dataset ne garde que les frames dont les 6 images et le lidar sont présents, et affiche
par exemple `v1.0-trainval / train : 2870/28130 frames disponibles sur disque, 71/700 scènes`.
La validation utilise les scènes de val officielles présentes dans les archives téléchargées.

Les métadonnées trainval décrivent les 850 scènes, quelle que soit la quantité de blobs
téléchargée. Le dataset en extrait donc une fois des infos compactes par frame, puis
`release_nuscenes()` libère l'objet NuScenes avant la création des workers du DataLoader :
sinon chacun des `NUM_WORKERS` processus pourrait finir par en dupliquer la mémoire.

## Fichiers (intégration manuelle)

**Nouveaux**

| Fichier | Rôle |
|---|---|
| `lss_det/config.py` | configuration unique + validation |
| `lss_det/engine.py` | boucles train / eval partagées |
| `lss_det/data/bev_augmentation.py` | BDA : rotation, échelle, miroirs du repère BEV |
| `lss_det/data/depth_targets.py` | labels de profondeur lidar par cellule de feature |
| `audit_dataset.py` | audit des cibles, portées, visibilité, couverture lidar |
| `print_history.py` | courbes d'apprentissage epoch par epoch depuis `history.pt` |
| `tests/fake_nuscenes.py` | faux nuScenes réaliste pour les tests |
| `tests/test_geometry_chain.py` | cohérence pixel ↔ frustum ↔ lidar ↔ BDA |
| `tests/test_dataset_fake_nuscenes.py` | dataset de bout en bout |
| `tests/test_model_targets_decode_metrics.py` | targets → décodage → métriques, forward/backward |
| `tests/test_audit_and_engine.py` | audit + moteur train/eval |
| `tests/test_heatmap_learnability.py` | garde-fou contre le piège « ReLU morte » de la heatmap |
| `tests/test_available_files.py` | partie de trainval : fichiers absents, NuScenes partagé puis libéré |
| `tests/__init__.py`, `lss_det/{data,decoding,losses,metrics,targets,visualization}/__init__.py` | packages |

**Réécrits** (remplacer le fichier entier) :
`lss_det/data/nuscenes_dataset.py`, `lss_det/data/transforms.py`, `lss_det/data/collate.py`,
`lss_det/models/camera_encoder.py`, `lss_det/models/lift.py`, `lss_det/models/geometry.py`,
`lss_det/models/lss_detector.py`, `lss_det/targets/centerpoint_targets.py`,
`lss_det/losses/detection_loss.py`, `lss_det/decoding/decode_boxes.py`,
`lss_det/metrics/detection_metrics.py`, `train.py`, `evaluate.py`, `README.md`.

**Modifié (une seule section)** : `lss_det/models/detection_head.py` — initialisation de la
conv finale de la heatmap (voir plus bas).

**Inchangés** : `lss_det/models/voxel_pooling.py`, `lss_det/models/bev_backbone.py`,
`tests/test_voxel_pooling.py`,
`tests/test_bev_backbone.py`, `tests/test_detection_head.py`.

**Supprimés / remplacés** :

| Ancien | Remplacé par |
|---|---|
| `train_sanity_check.py` | `OVERFIT_NUM_SAMPLES` dans `config.py` |
| `inspect_dataset.py` | `audit_dataset.py` |
| `test_full_forward.py` | `tests/test_model_targets_decode_metrics.py` |
| `configs/nuscenes.yaml`, `infer.py` (vides) | `lss_det/config.py` |
| `tests/test_lift.py`, `test_geometry.py`, `test_camera_encoder.py`, `test_lss_detector.py`, `test_centerpoint_targets.py`, `test_detection_loss.py`, `test_decode_boxes.py` | nouveaux tests (ancienne API) |

Penser à supprimer les `__pycache__` après copie : un `.pyc` périmé peut masquer un changement.

## Changements v1 → v2

| | v1 | v2 |
|---|---|---|
| Image → features | 128×352 → 8×22 | 256×704 → 16×44 |
| Frustum | `linspace(0, W-1, Wf)` (biais ±7,5 px) | centres de cellule `16j+7.5`, centres de bin |
| Profondeur | 4–45 m, 41 bins, non supervisée | 1–60 m, 59 bins, supervisée par lidar (BCE, poids 3) |
| Lift | conv 1×1 partagée | DepthNet camera-aware (K effectif + extrinsèques) + 2 blocs résiduels |
| BEV | ±50 m @ 0,5 m (200²), z ±10 | ±51,2 m @ 0,8 m (128²), z ∈ [-5, 3] |
| Augmentation | ColorJitter | resize / crop / flip / rotation image + BDA |
| GT | toutes dans ±50 m | sans les boîtes à 0 point lidar+radar, portée par classe (50 / 40 m) |
| Régression | L1 en mètres, 4 termes × 1 | log-dimensions, pondération CenterPoint 0,25 |
| NMS | rayon 1 m pour tout | rayon par classe |
| Init conv finale heatmap | std 0,001 (piège ReLU morte) | init PyTorch par défaut, biais −2,19 |
| EfficientNet | `_conv_head` / `_fc` calculés ou comptés pour rien | supprimés, LR backbone × 0,1 |
| Optimisation | lr constant 1e-4, wd 1e-4, batch 1 | AdamW 2e-4, wd 1e-2, warmup + cosinus, batch effectif 8 |
| Sélection | val loss | mAP (style nuScenes) |

## Correction de la tête heatmap

Avec `std=0.001` sur la conv finale de la heatmap, le premier pas d'Adam rend tous ses poids
négatifs (les ~100 000 cellules de fond dominent, les features post-ReLU sont ≥ 0). Le réseau
éteint alors les features aux centres d'objets et la probabilité y reste figée à
sigmoid(−2,19) = 0,10. Mesuré sur BEV 128×128 avec un signal objet évident, Adam lr 2e-4 :

| init | GTp après 20 / 40 / 60 / 80 / 100 pas |
|---|---|
| std 0,001 (v1) | 0,09 / 0,10 / 0,10 / 0,10 / 0,10 |
| défaut PyTorch (v2) | 0,17 / 0,28 / 0,39 / 0,53 / 0,62 |

`tests/test_heatmap_learnability.py` échoue avec l'ancienne init et passe avec la nouvelle.

## Lire les logs

- **mAP, AP@{0.5,1,2,4} m, ATE/ASE/AOE** : même algorithme d'appariement que le devkit
  nuScenes (sans vitesse ni attributs, donc pas de NDS officiel).
- **rappel @2 m par distance** : montre si les échecs sont lointains (résolution, profondeur)
  ou proches (autre chose).
- **dAcc / dRel** : précision de la profondeur prédite contre le lidar (argmax à ±1 bin ;
  erreur relative de l'espérance). Comparer TRAIN et VAL :
  - dAcc val proche de train, mais mAP val faible → le verrou est côté BEV / tête ;
  - dAcc val s'effondre → le verrou est côté encodeur / profondeur.
- **GTp** : probabilité heatmap aux centres GT, diagnostic uniquement.
- La loss de validation peut remonter alors que le mAP progresse (modèle plus confiant) :
  ne pas l'utiliser pour choisir un checkpoint.

## Ablations recommandées (une à la fois, mesurées en mAP)

`DEPTH_WEIGHT = 0` · BDA neutre (`rot_lim=(0,0)`, `scale_lim=(1,1)`, probas de flip à 0) · `IMG_AUG` neutre ·
`FILTER_EMPTY_BOXES = False` · `BACKBONE_LR_MULT = 1.0` · `IMAGE_SIZE = (128, 352)`.
C'est ce qui attribue réellement le gain à chaque cause.
