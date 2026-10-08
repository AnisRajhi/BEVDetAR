#!/usr/bin/env python3
"""
Prétraitement / augmentation image pour LSS.

Pipeline (même ordre que LSS / BEVDet) :

    image originale
        -> resize   (aléatoire en train)
        -> crop     (aléatoire en train)
        -> flip H   (aléatoire en train)
        -> rotation (aléatoire en train)
        -> jitter photométrique (train)
        -> tensor + normalisation ImageNet

Chaque opération géométrique est composée dans (post_rot, post_trans) :

    [u', v'] = post_rot[:2,:2] @ [u, v] + post_trans[:2]

où (u, v) sont les coordonnées pixel de l'image ORIGINALE, en convention
"centre de pixel entier" (celle des intrinsèques nuScenes/OpenCV).

Geometry inverse exactement cette relation et les labels de profondeur
lidar sont projetés avec cette même relation : l'augmentation image ne
casse donc jamais la cohérence géométrique. C'est ce qui permet
d'augmenter la diversité géométrique sans toucher aux extrinsèques.

Détail de convention : PIL travaille en coordonnées continues (pixel i
couvre [i, i+1)). On compose donc tout en continu, puis on convertit
une seule fois vers la convention "centre entier" à la fin.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
_FLIP_LR = getattr(Image, "Transpose", Image).FLIP_LEFT_RIGHT


def rot2d(angle_deg: float) -> np.ndarray:
    """
    Matrice 2D équivalente à PIL.Image.rotate(angle_deg) :
    rotation anti-horaire à l'écran, axe v orienté vers le bas.
    """
    h = angle_deg / 180.0 * math.pi
    c, s = math.cos(h), math.sin(h)
    return np.array([[c, s], [-s, c]], dtype=np.float64)


class LSSImageAugmentation:
    def __init__(
        self,
        final_dim: Tuple[int, int],
        training: bool,
        aug: Optional[Dict] = None,
        val_bot_pct: float = 0.0,
        normalize: bool = True,
    ):
        self.fH, self.fW = int(final_dim[0]), int(final_dim[1])
        self.training = bool(training)
        self.aug = dict(aug or {})
        self.val_bot_pct = float(val_bot_pct)
        self.normalize = bool(normalize)

        self.color_jitter = None
        if self.training and any(
            self.aug.get(k, 0.0) > 0 for k in ("brightness", "contrast", "saturation", "hue")
        ):
            self.color_jitter = ColorJitter(
                brightness=self.aug.get("brightness", 0.0),
                contrast=self.aug.get("contrast", 0.0),
                saturation=self.aug.get("saturation", 0.0),
                hue=self.aug.get("hue", 0.0),
            )

    # --------------------------------------------------------------
    # Tirage des paramètres
    # --------------------------------------------------------------

    def sample_params(self, W: int, H: int, rng: np.random.Generator, force_flip: Optional[bool] = None) -> Dict:
        """
        force_flip : None = tirage indépendant (IMG_AUG["rand_flip"]) ;
                     True/False = imposé par le miroir cohérent de la frame (dataset).
        """
        base = max(self.fW / W, self.fH / H)

        if self.training:
            resize = base * rng.uniform(*self.aug.get("resize_lim", (1.0, 1.0)))
            bot_pct = rng.uniform(*self.aug.get("bot_pct_lim", (0.0, 0.0)))
            if force_flip is not None:
                flip = bool(force_flip)
            else:
                flip = bool(self.aug.get("rand_flip", False)) and bool(rng.random() < 0.5)
            rotate = float(rng.uniform(*self.aug.get("rot_lim", (0.0, 0.0))))
        else:
            resize = base
            bot_pct = self.val_bot_pct
            flip = False
            rotate = 0.0

        newW = int(round(W * resize))
        newH = int(round(H * resize))

        # Crop vertical : on garde le bas de l'image (là où sont les objets),
        # en rognant bot_pct en bas. Borné pour rester dans l'image.
        max_crop_h = newH - self.fH
        crop_h = int((1.0 - bot_pct) * newH) - self.fH
        crop_h = int(np.clip(crop_h, min(0, max_crop_h), max(0, max_crop_h)))

        max_crop_w = newW - self.fW
        if self.training and max_crop_w > 0:
            crop_w = int(rng.uniform(0, max_crop_w))
        else:
            crop_w = int(max_crop_w / 2)

        return {
            "resize_dims": (newW, newH),
            "crop": (crop_w, crop_h, crop_w + self.fW, crop_h + self.fH),
            "flip": flip,
            "rotate": rotate,
        }

    # --------------------------------------------------------------
    # Transformation pixel original -> pixel augmenté
    # --------------------------------------------------------------

    @staticmethod
    def compute_post_transform(original_size: Tuple[int, int], params: Dict):
        """
        original_size = (W, H). Retourne (post_rot [3,3], post_trans [3]).
        """
        W, H = original_size
        newW, newH = params["resize_dims"]
        crop = params["crop"]
        cw, ch = crop[2] - crop[0], crop[3] - crop[1]

        # --- composition en coordonnées continues ---
        R = np.diag([newW / W, newH / H]).astype(np.float64)
        t = -np.array(crop[:2], dtype=np.float64)

        if params["flip"]:
            A = np.array([[-1.0, 0.0], [0.0, 1.0]])
            b = np.array([float(cw), 0.0])
            R, t = A @ R, A @ t + b

        if params["rotate"] != 0.0:
            A = rot2d(params["rotate"])
            c = np.array([cw / 2.0, ch / 2.0])   # centre de rotation PIL
            R, t = A @ R, A @ t + (c - A @ c)

        # --- conversion vers la convention "centre de pixel entier" ---
        # u_c = u + 0.5 ;  u'_c = R u_c + t ;  u' = u'_c - 0.5
        half = np.array([0.5, 0.5])
        t = t + R @ half - half

        post_rot = torch.eye(3, dtype=torch.float32)
        post_rot[:2, :2] = torch.from_numpy(R).float()
        post_trans = torch.zeros(3, dtype=torch.float32)
        post_trans[:2] = torch.from_numpy(t).float()
        return post_rot, post_trans

    @staticmethod
    def apply_to_image(image: Image.Image, params: Dict) -> Image.Image:
        image = image.resize(params["resize_dims"], resample=_BILINEAR)
        image = image.crop(params["crop"])
        if params["flip"]:
            image = image.transpose(_FLIP_LR)
        if params["rotate"] != 0.0:
            image = image.rotate(params["rotate"], resample=_BILINEAR)
        return image

    def geometric(self, image: Image.Image, params: Dict) -> Image.Image:
        """
        Décodage + transformations géométriques.

        v2.3 : image.draft() demande au décodeur JPEG de décoder directement
        à 1/2 (ou 1/4…) de la taille, sans descendre sous la taille du
        resize. Le décodage et le resize deviennent plusieurs fois plus
        rapides (le CPU était le goulot : GPU ~20 %). La géométrie ne change
        pas : la réduction JPEG est un facteur exact, et le resize qui suit
        vise les mêmes dimensions finales (testé à moins de 0,2 px près).
        Sans effet sur les images non JPEG.
        """
        image.draft("RGB", params["resize_dims"])
        return self.apply_to_image(image.convert("RGB"), params)

    def __call__(self, image: Image.Image, rng: np.random.Generator, force_flip: Optional[bool] = None):
        original_size = image.size          # lue dans l'en-tête, avant décodage
        params = self.sample_params(original_size[0], original_size[1], rng, force_flip)
        post_rot, post_trans = self.compute_post_transform(original_size, params)
        image = self.geometric(image, params)

        if self.training and self.color_jitter is not None:
            image = self.color_jitter(image)

        tensor = TF.to_tensor(image)
        if self.normalize:
            tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        return tensor, post_rot, post_trans
