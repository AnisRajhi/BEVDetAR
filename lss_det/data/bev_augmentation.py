#!/usr/bin/env python3
"""
BEV Data Augmentation (BDA, BEVDet).

Pourquoi c'est central pour LSS
-------------------------------
Les augmentations image (resize/crop/flip/rot) régularisent le
CameraEncoder, mais le BEVBackbone et la DetectionHead voient toujours
la MÊME disposition spatiale : ego au centre, route orientée de la même
façon, objets aux mêmes endroits d'une frame à l'autre de la scène.
Avec ~8 scènes, le BEV encoder peut apprendre "où sont les voitures dans
la scène X" au lieu de "à quoi ressemble une voiture en BEV".

BEVDet montre que sans augmentation dans l'espace BEV, le BEV encoder
sur-apprend massivement. La BDA tourne / met à l'échelle / miroite
le monde ENTIER :

    P_bev = bda @ P_ego

appliqué à la fois :
    - aux points du frustum (dans Geometry),
    - aux boîtes GT (ici).

Les images ne changent pas, la profondeur caméra ne change pas : on
observe simplement la même scène dans un repère BEV tourné/miroité.
C'est exact et gratuit (aucun pixel rééchantillonné).
"""

import math
from typing import Dict, Tuple

import numpy as np
import torch


def sample_bda(rng: np.random.Generator, cfg: Dict, training: bool) -> Tuple[torch.Tensor, Dict]:
    if not training:
        return torch.eye(3, dtype=torch.float32), {
            "rot_deg": 0.0, "scale": 1.0, "flip_x": False, "flip_y": False,
        }

    params = {
        "rot_deg": float(rng.uniform(*cfg["rot_lim"])),
        "scale": float(rng.uniform(*cfg["scale_lim"])),
        "flip_x": bool(rng.random() < cfg["flip_x_prob"]),
        "flip_y": bool(rng.random() < cfg["flip_y_prob"]),
    }
    return build_bda_matrix(**params), params


def build_bda_matrix(rot_deg: float, scale: float, flip_x: bool, flip_y: bool) -> torch.Tensor:
    a = rot_deg / 180.0 * math.pi
    c, s = math.cos(a), math.sin(a)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    scl = np.eye(3) * scale
    flip = np.diag([-1.0 if flip_x else 1.0, -1.0 if flip_y else 1.0, 1.0])
    return torch.from_numpy(flip @ scl @ rot).float()


def apply_bda_to_boxes(boxes: torch.Tensor, bda: torch.Tensor) -> torch.Tensor:
    """
    boxes : [M, 7] = [x, y, z, l, w, h, yaw]
    bda   : [3, 3] = flip @ (scale * rot_z)

    Le centre est transformé par bda. Le cap est recalculé en
    transformant le vecteur direction (cos yaw, sin yaw, 0) : cela gère
    rotation ET miroir sans cas particulier. Les dimensions sont
    multipliées par l'échelle (isotrope).
    """
    if boxes.numel() == 0:
        return boxes.clone()

    out = boxes.clone()
    out[:, 0:3] = boxes[:, 0:3] @ bda.T

    scale = torch.linalg.norm(bda[:, 0]).item()
    out[:, 3:6] = boxes[:, 3:6] * scale

    heading = torch.stack(
        [torch.cos(boxes[:, 6]), torch.sin(boxes[:, 6]), torch.zeros_like(boxes[:, 6])], dim=1
    ) @ bda.T
    out[:, 6] = torch.atan2(heading[:, 1], heading[:, 0])
    return out
