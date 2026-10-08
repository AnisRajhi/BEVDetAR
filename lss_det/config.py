#!/usr/bin/env python3
"""
Configuration unique du projet (paramètres en dur, volontairement).

Tous les modules (dataset, modèle, targets, décodage, métriques) lisent
LEURS paramètres ici. Avant, la même grille BEV était recopiée dans
train.py, dans le TargetBuilder, dans le Decoder et dans le modèle :
une incohérence silencieuse était possible.

validate() vérifie les RELATIONS entre paramètres (plage de profondeur
vs portée des GT, taille image vs feature map, etc.) et lève une erreur
explicite plutôt que de laisser un mode de détection devenir
inatteignable sans bruit.
"""

import math

# ==================================================================
# Données
# ==================================================================

DATAROOT = "~/datasets/nuscenes"
VERSION = "v1.0-trainval"

CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]

CLASSES = [
    "car",
    "truck",
    "bus",
    "pedestrian",
    "bicycle",
    "motorcycle",
]

# Portée d'évaluation officielle nuScenes (distance euclidienne XY
# dans le repère ego). Les GT au-delà sont retirées EN TRAIN ET EN VAL :
# on ne demande plus au réseau des objets qu'aucune métrique ne compte
# et que la géométrie ne peut souvent pas atteindre.
CLASS_RANGE = {
    "car": 50.0,
    "truck": 50.0,
    "bus": 50.0,
    "pedestrian": 40.0,
    "bicycle": 40.0,
    "motorcycle": 40.0,
}

# Filtre officiel nuScenes : une annotation avec 0 point lidar ET
# 0 point radar est un objet totalement occulté. Le demander à un
# détecteur caméra = lui demander de mémoriser la scène.
FILTER_EMPTY_BOXES = True

# Visibilité nuScenes : 1 = 0-40 %, 2 = 40-60 %, 3 = 60-80 %, 4 = 80-100 %.
# 1 = on garde tout (comportement de l'évaluation officielle).
MIN_VISIBILITY_LEVEL = 1

# Entraîner sur une PARTIE de nuScenes trainval :
#   1. extraire les métadonnées v1.0-trainval + une ou plusieurs archives
#      de blobs dans DATAROOT (dossiers samples/CAM_*, samples/LIDAR_TOP) ;
#   2. VERSION = "v1.0-trainval" (splits officiels : 700 scènes train, 150 val).
# Seules les frames dont les 6 images et le lidar existent sur le disque
# sont gardées ; le nombre retenu est affiché à la construction du dataset.
REQUIRE_AVAILABLE_FILES = True

# Miroir COHÉRENT (v2.2), entraînement uniquement : un seul tirage par frame.
# Avec la probabilité MIRROR_PROB, les 6 images sont retournées
# horizontalement ET le monde BEV est miroité ; sinon, ni l'un ni l'autre.
# Cela équivaut à filmer un monde miroité avec des caméras normales :
# l'apparence des objets reste cohérente avec leur orientation cible.
# Quand MIRROR_PROB n'est pas None, IMG_AUG["rand_flip"] et
# BDA["flip_x_prob"/"flip_y_prob"] sont IGNORÉS.
# None = ancien comportement (miroirs indépendants : bloque l'orientation).
MIRROR_PROB = 0.5

# ==================================================================
# Image / caméra
# ==================================================================

ORIGINAL_IMAGE_SIZE = (900, 1600)  # (H, W) nuScenes
IMAGE_SIZE = (256, 704)            # (H, W) entrée réseau
DOWNSAMPLE = 16                    # stride de la feature map caméra
FEATURE_SIZE = (IMAGE_SIZE[0] // DOWNSAMPLE, IMAGE_SIZE[1] // DOWNSAMPLE)

# Repli si la mémoire GPU est insuffisante :
#   IMAGE_SIZE = (128, 352)  -> FEATURE_SIZE = (8, 22)

# Augmentation image (géométrique + photométrique), TRAIN uniquement.
# Toutes les transformations géométriques sont propagées dans
# post_rot / post_trans, donc la géométrie LSS reste exacte.
IMG_AUG = {
    "resize_lim": (0.90, 1.10),    # facteur relatif au resize de base
    "bot_pct_lim": (0.00, 0.22),   # fraction rognée en bas
    "rot_lim": (-5.4, 5.4),        # degrés
    "rand_flip": False,
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2,
    "hue": 0.05,
}
VAL_BOT_PCT = 0.0

# ==================================================================
# Profondeur (frustum)
# ==================================================================

# [min, max, pas] en mètres, profondeur = Z optique caméra.
DEPTH_BOUND = (1.0, 60.0, 1.0)
NUM_DEPTH_BINS = int(round((DEPTH_BOUND[1] - DEPTH_BOUND[0]) / DEPTH_BOUND[2]))

# ==================================================================
# Grille BEV
# ==================================================================

XBOUND = (-51.2, 51.2, 0.8)
YBOUND = (-51.2, 51.2, 0.8)
ZBOUND = (-5.0, 3.0, 8.0)

NX = int(round((XBOUND[1] - XBOUND[0]) / XBOUND[2]))
NY = int(round((YBOUND[1] - YBOUND[0]) / YBOUND[2]))

# BEV Data Augmentation (BDA), TRAIN uniquement.
BDA = {
    "rot_lim": (-22.5, 22.5),      # degrés
    "scale_lim": (0.95, 1.05),
    "flip_x_prob": 0.0,
    "flip_y_prob": 0.0,
}

# ==================================================================
# Modèle
# ==================================================================

CAMERA_PRETRAINED = True
CAMERA_FEATURE_CHANNELS = 256
DEPTHNET_MID_CHANNELS = 256
CONTEXT_CHANNELS = 64
BEV_CHANNELS = 128

# ==================================================================
# Targets / loss
# ==================================================================

MAX_OBJECTS = 500
GAUSSIAN_OVERLAP = 0.1
MIN_RADIUS = 2

HEATMAP_WEIGHT = 1.0
BBOX_WEIGHT = 0.25          # CenterPoint
# poids par composante : [dx, dy, z, log_l, log_w, log_h, sin, cos]
BBOX_CODE_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
DEPTH_WEIGHT = 3.0          # BEVDepth

# ==================================================================
# Décodage
# ==================================================================

DECODER_TOP_K = 300
DECODER_SCORE_THRESHOLD = 0.05
# Rayon de circle-NMS par classe (mètres), ordre de grandeur CenterPoint.
NMS_RADIUS = {
    "car": 2.0,
    "truck": 2.5,
    "bus": 3.5,
    "pedestrian": 0.5,
    "bicycle": 0.6,
    "motorcycle": 0.6,
}

# ==================================================================
# Entraînement
# ==================================================================

SEED = 42
EPOCHS = 10
BATCH_SIZE = 1
ACCUM_STEPS = 8               # batch effectif = BATCH_SIZE * ACCUM_STEPS
NUM_WORKERS = 4
LEARNING_RATE = 2e-4
BACKBONE_LR_MULT = 0.1        # EfficientNet : LR x 0.1
FREEZE_BACKBONE_BLOCKS = 0    # >0 : gèle le stem + les N premiers MBConv
WEIGHT_DECAY = 1e-2           # AdamW : 1e-4 était quasiment nul (voir doc)
WARMUP_STEPS = 500            # en pas d'optimiseur
MIN_LR_RATIO = 1e-3
GRAD_CLIP = 35.0
EVAL_EVERY = 2
# >0 : sanity check d'overfit sur les N premières frames train,
# sans augmentation, évaluées sur ces mêmes frames. Dans ce mode,
# ACCUM_STEPS est forcé à 1 et on utilise OVERFIT_EPOCHS / OVERFIT_WARMUP_STEPS
# (sinon 10 frames / 8 = 1 pas par epoch : on ne sortirait jamais du warmup).
OVERFIT_NUM_SAMPLES = 0
OVERFIT_EPOCHS = 150
OVERFIT_WARMUP_STEPS = 50
OVERFIT_EVAL_EVERY = 25
CHECKPOINT_DIR = "checkpoints_full"


# ==================================================================
# Validation des relations entre paramètres
# ==================================================================

def validate():
    errors = []

    H, W = IMAGE_SIZE
    if H % DOWNSAMPLE or W % DOWNSAMPLE:
        errors.append(f"IMAGE_SIZE {IMAGE_SIZE} doit être divisible par {DOWNSAMPLE}.")

    if abs(NX * XBOUND[2] - (XBOUND[1] - XBOUND[0])) > 1e-6:
        errors.append("XBOUND : l'étendue n'est pas un multiple de la résolution.")
    if abs(NY * YBOUND[2] - (YBOUND[1] - YBOUND[0])) > 1e-6:
        errors.append("YBOUND : l'étendue n'est pas un multiple de la résolution.")

    max_range = max(CLASS_RANGE.values())
    bev_half = min(abs(XBOUND[0]), XBOUND[1], abs(YBOUND[0]), YBOUND[1])
    if bev_half < max_range:
        errors.append(
            f"La grille BEV (±{bev_half} m) ne couvre pas la portée max des GT ({max_range} m)."
        )

    # Profondeur caméra <= distance euclidienne + offset caméra (~2 m).
    # Pour qu'un objet à la portée max soit atteignable par au moins un
    # bin, il faut DEPTH_BOUND[1] >= max_range + marge.
    if DEPTH_BOUND[1] < max_range + 2.0:
        errors.append(
            f"DEPTH_BOUND max={DEPTH_BOUND[1]} < portée GT {max_range} + 2 m : "
            "des objets cibles seraient inatteignables par le Lift."
        )

    if set(CLASS_RANGE) != set(CLASSES) or set(NMS_RADIUS) != set(CLASSES):
        errors.append("CLASS_RANGE / NMS_RADIUS doivent couvrir exactement CLASSES.")

    if len(BBOX_CODE_WEIGHTS) != 8:
        errors.append("BBOX_CODE_WEIGHTS doit contenir 8 valeurs.")

    if MIRROR_PROB is not None and not 0.0 <= MIRROR_PROB <= 1.0:
        errors.append("MIRROR_PROB doit être None ou dans [0, 1].")

    if ACCUM_STEPS < 1 or BATCH_SIZE < 1:
        errors.append("BATCH_SIZE et ACCUM_STEPS doivent être >= 1.")

    if errors:
        raise ValueError("Configuration invalide :\n  - " + "\n  - ".join(errors))


def summary() -> str:
    Hf, Wf = FEATURE_SIZE
    return (
        f"image {IMAGE_SIZE} -> features {FEATURE_SIZE} "
        f"({Hf * Wf} cellules/caméra) | "
        f"depth {DEPTH_BOUND} -> {NUM_DEPTH_BINS} bins | "
        f"BEV {NX}x{NY} @ {XBOUND[2]} m | "
        f"z {ZBOUND[:2]}"
    )


validate()
