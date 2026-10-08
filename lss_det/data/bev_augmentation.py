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
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def sample_bda(rng: np.random.Generator, cfg: Dict, training: bool,
               mirror: Optional[bool] = None) -> Tuple[torch.Tensor, Dict]:
    """
    mirror : None = miroirs X et Y tirés indépendamment (cfg["flip_*_prob"]).
             True/False = miroir cohérent imposé par la frame : miroir Y seul
             (gauche <-> droite) si mirror, rien sinon.

    v2.3.1 — mesuré sur la partie 1 de trainval : l'orientation n'apprend que
    si l'avant de l'ego reste orienté vers +x dans la grille BEV. Un miroir X
    ou un retournement de 180° (présents dans ~50 % des frames en v2.1 et en
    v2.3) inverse le cap de toutes les cibles : loss rot figée à 1,21, avec
    ou sans cohérence image/monde. Sans ces retournements : 1,21 -> 1,11 en
    5 epochs. Le miroir Y garde l'avant vers +x.

    Pourquoi la parité doit suivre le retournement des images
    ---------------------------------------------------------
    Retourner une image change l'apparence d'un objet en son image miroir
    (une voiture « vers la droite » paraît aller « vers la gauche »).
    Si le monde BEV n'est pas miroité en même temps, la cible d'orientation
    contredit l'apparence : sur la moitié des exemples, l'indice d'orientation
    le plus direct devient faux et la tête rot n'apprend rien (mesuré :
    loss rot figée à 1,21 pendant 24 epochs). Retourner TOUTES les images ET
    miroiter le monde ensemble équivaut à filmer un monde miroité avec une
    caméra normale : tout reste cohérent.
    """
    if not training:
        return torch.eye(3, dtype=torch.float32), {
            "rot_deg": 0.0, "scale": 1.0, "flip_x": False, "flip_y": False,
        }

    params = {
        "rot_deg": float(rng.uniform(*cfg["rot_lim"])),
        "scale": float(rng.uniform(*cfg["scale_lim"])),
        # Miroir cohérent : miroir GAUCHE <-> DROITE uniquement (y -> -y). Jamais de
        # miroir X ni de 180° : ils retournent l'axe « avant » de l'ego dans la
        # grille BEV et inversent le cap de toutes les cibles (v2.3.1).
        "flip_x": bool(rng.random() < cfg["flip_x_prob"]) if mirror is None else False,
        "flip_y": None,
    }
    params["flip_y"] = (bool(rng.random() < cfg["flip_y_prob"]) if mirror is None
                        else params["flip_x"] != bool(mirror))
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
