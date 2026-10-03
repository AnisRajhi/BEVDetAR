#!/usr/bin/env python3

from typing import Tuple

import torch

from PIL import Image

from torchvision.transforms import functional as TF
from torchvision.transforms import ColorJitter


class LSSImageTransform:
    """
    Image preprocessing pour LSS.

    Pipeline:

        image originale
            ↓
        resize
            ↓
        crop
            ↓
        augmentation photométrique OPTIONNELLE
            ↓
        tensor
            ↓
        ImageNet normalization

    IMPORTANT
    ---------

    Les augmentations photométriques ne changent PAS:

        - la géométrie
        - les intrinsics
        - les extrinsics
        - post_rot
        - post_trans

    Elles changent uniquement l'apparence RGB.
    """

    def __init__(
        self,
        final_dim: Tuple[int, int] = (
            128,
            352,
        ),
        bottom_crop_pct: float = 0.11,
        normalize: bool = True,

        # ======================================================
        # Data augmentation
        # ======================================================

        training: bool = False,

        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
    ):
        self.final_h = int(
            final_dim[0]
        )

        self.final_w = int(
            final_dim[1]
        )

        self.bottom_crop_pct = float(
            bottom_crop_pct
        )

        self.normalize = bool(
            normalize
        )

        self.training = bool(
            training
        )

        # ======================================================
        # ImageNet normalization
        # ======================================================

        self.mean = [
            0.485,
            0.456,
            0.406,
        ]

        self.std = [
            0.229,
            0.224,
            0.225,
        ]

        # ======================================================
        # Photometric augmentation
        # ======================================================
        #
        # Par exemple:
        #
        # brightness = 0.2
        #
        # signifie environ:
        #
        # [0.8, 1.2]
        #
        # autour de la luminosité originale.
        #
        # ======================================================

        self.color_jitter = ColorJitter(

            brightness=brightness,

            contrast=contrast,

            saturation=saturation,

            hue=hue,
        )

    def __call__(
        self,
        image: Image.Image,
    ):

        image = image.convert(
            "RGB"
        )

        original_w, original_h = (
            image.size
        )

        # ======================================================
        # 1. Resize
        # ======================================================

        resize = max(

            self.final_w
            / original_w,

            self.final_h
            / original_h,
        )

        resized_w = int(
            round(
                original_w
                * resize
            )
        )

        resized_h = int(
            round(
                original_h
                * resize
            )
        )

        if hasattr(
            Image,
            "Resampling",
        ):

            bilinear = (
                Image.Resampling.BILINEAR
            )

        else:

            bilinear = (
                Image.BILINEAR
            )

        image = image.resize(

            (
                resized_w,
                resized_h,
            ),

            resample=bilinear,
        )

        # Facteurs exacts après arrondi.

        scale_x = (
            resized_w
            / original_w
        )

        scale_y = (
            resized_h
            / original_h
        )

        # ======================================================
        # 2. Crop
        # ======================================================

        max_crop_x = max(

            0,

            resized_w
            - self.final_w,
        )

        crop_x = (
            max_crop_x // 2
        )

        crop_y = int(
            (
                1.0
                - self.bottom_crop_pct
            )
            * resized_h
        ) - self.final_h

        max_crop_y = max(

            0,

            resized_h
            - self.final_h,
        )

        crop_y = max(

            0,

            min(
                crop_y,
                max_crop_y,
            ),
        )

        image = image.crop(
            (
                crop_x,

                crop_y,

                crop_x
                + self.final_w,

                crop_y
                + self.final_h,
            )
        )

        # ======================================================
        # 3. Geometry post transform
        # ======================================================
        #
        # u_aug =
        #
        #     scale_x * u_original
        #     - crop_x
        #
        #
        # v_aug =
        #
        #     scale_y * v_original
        #     - crop_y
        #
        # ======================================================

        post_rot = torch.eye(

            3,

            dtype=torch.float32,
        )

        post_rot[
            0,
            0,
        ] = scale_x

        post_rot[
            1,
            1,
        ] = scale_y

        post_trans = torch.tensor(

            [
                -float(
                    crop_x
                ),

                -float(
                    crop_y
                ),

                0.0,
            ],

            dtype=torch.float32,
        )

        # ======================================================
        # 4. Photometric augmentation
        # ======================================================
        #
        # TRAIN uniquement.
        #
        # Rien n'est modifié dans post_rot / post_trans
        # puisqu'on ne déplace aucun pixel géométriquement.
        #
        # ======================================================

        if self.training:

            image = self.color_jitter(
                image
            )

        # ======================================================
        # 5. PIL -> Tensor
        # ======================================================

        image = TF.to_tensor(
            image
        )

        # ======================================================
        # 6. Normalize
        # ======================================================

        if self.normalize:

            image = TF.normalize(

                image,

                mean=self.mean,

                std=self.std,
            )

        return (
            image,

            post_rot,

            post_trans,
        )