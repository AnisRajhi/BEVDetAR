#!/usr/bin/env python3

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from lss_det.models.camera_encoder import CameraEncoder
from lss_det.models.lift import Lift
from lss_det.models.geometry import Geometry
from lss_det.models.voxel_pooling import VoxelPooling
from lss_det.models.bev_backbone import BEVBackbone
from lss_det.models.detection_head import DetectionHead


class LSSDetector(nn.Module):
    """
    Lift-Splat-Shoot based 3D detector.

    Pipeline:

        RGB images
            ↓
        CameraEncoder
            ↓
        Lift
            ↓
        Geometry
            ↓
        VoxelPooling / SPLAT
            ↓
        BEVBackbone
            ↓
        DetectionHead


    ==============================================================
    INPUTS
    ==============================================================

    images:
        [B,N,3,H,W]

    intrins:
        [B,N,3,3]

    rots:
        [B,N,3,3]

        camera -> ego rotation

    trans:
        [B,N,3]

        camera -> ego translation

    post_rots:
        [B,N,3,3]

    post_trans:
        [B,N,3]


    ==============================================================
    OUTPUT
    ==============================================================

    {
        "heatmap":   [B,K,Nx,Ny],
        "reg":       [B,2,Nx,Ny],
        "center_z":  [B,1,Nx,Ny],
        "dim":       [B,3,Nx,Ny],
        "rot":       [B,2,Nx,Ny],
    }
    """

    def __init__(
        self,
        num_classes: int,

        image_size: Tuple[int, int] = (
            128,
            352,
        ),

        feature_size: Tuple[int, int] = (
            8,
            22,
        ),

        depth_bound: Tuple[float, float, float] = (
            4.0,
            45.0,
            1.0,
        ),

        xbound: Tuple[float, float, float] = (
            -50.0,
            50.0,
            0.5,
        ),

        ybound: Tuple[float, float, float] = (
            -50.0,
            50.0,
            0.5,
        ),

        zbound: Tuple[float, float, float] = (
            -10.0,
            10.0,
            20.0,
        ),

        camera_feature_channels: int = 512,
        context_channels: int = 64,
        bev_channels: int = 128,

        camera_pretrained: bool = False,
        camera_weights_path: Optional[str] = None,

        return_intermediates: bool = False,
    ):
        super().__init__()

        self.num_classes = int(
            num_classes
        )

        self.image_size = tuple(
            image_size
        )

        self.feature_size = tuple(
            feature_size
        )

        self.depth_bound = tuple(
            depth_bound
        )

        self.xbound = tuple(
            xbound
        )

        self.ybound = tuple(
            ybound
        )

        self.zbound = tuple(
            zbound
        )

        self.camera_feature_channels = int(
            camera_feature_channels
        )

        self.context_channels = int(
            context_channels
        )

        self.bev_channels = int(
            bev_channels
        )

        self.return_intermediates = bool(
            return_intermediates
        )

        # ==========================================================
        # Number of depth bins
        # ==========================================================

        self.num_depth_bins = int(
            round(
                (
                    depth_bound[1]
                    - depth_bound[0]
                )
                / depth_bound[2]
            )
        )

        # Avec:
        #
        # [4,45,1]
        #
        # D = 41

        # ==========================================================
        # 1. Camera Encoder
        # ==========================================================
        #
        # [B,N,3,128,352]
        #
        # →
        #
        # [B,N,512,8,22]
        #

        self.camera_encoder = CameraEncoder(
            pretrained=camera_pretrained,
            weights_path=camera_weights_path,
        )

        # ==========================================================
        # 2. Lift
        # ==========================================================
        #
        # [B,N,512,8,22]
        #
        # →
        #
        # depth:
        # [B,N,41,8,22]
        #
        # context:
        # [B,N,64,8,22]
        #
        # lifted:
        # [B,N,41,8,22,64]
        #

        self.lift = Lift(
            in_channels=self.camera_feature_channels,
            num_depth_bins=self.num_depth_bins,
            context_channels=self.context_channels,
        )

        # ==========================================================
        # 3. Geometry
        # ==========================================================
        #
        # calibration
        #
        # →
        #
        # [B,N,41,8,22,3]
        #

        self.geometry = Geometry(
            image_size=self.image_size,
            feature_size=self.feature_size,
            depth_bound=self.depth_bound,
        )

        # ==========================================================
        # 4. Voxel Pooling
        # ==========================================================
        #
        # lifted + XYZ
        #
        # →
        #
        # [B,64,200,200]
        #

        self.voxel_pooling = VoxelPooling(
            xbound=self.xbound,
            ybound=self.ybound,
            zbound=self.zbound,
            collapse_z=True,
        )

        # ==========================================================
        # 5. BEV Backbone
        # ==========================================================
        #
        # [B,64,200,200]
        #
        # →
        #
        # [B,128,200,200]
        #

        self.bev_backbone = BEVBackbone(
            in_channels=self.context_channels,
            out_channels=self.bev_channels,
        )

        # ==========================================================
        # 6. Detection Head
        # ==========================================================

        self.detection_head = DetectionHead(
            in_channels=self.bev_channels,
            shared_channels=self.bev_channels,
            head_channels=64,
            num_classes=self.num_classes,
        )

    # ==================================================================
    # Input checking
    # ==================================================================

    def _check_inputs(
        self,
        images: torch.Tensor,
        intrins: torch.Tensor,
        rots: torch.Tensor,
        trans: torch.Tensor,
        post_rots: torch.Tensor,
        post_trans: torch.Tensor,
    ):

        if images.ndim != 5:

            raise ValueError(
                "images must have shape "
                "[B,N,3,H,W]. "
                f"Received {tuple(images.shape)}"
            )

        B, N, C, H, W = images.shape

        if C != 3:

            raise ValueError(
                "Images must contain 3 RGB channels."
            )

        if (H, W) != self.image_size:

            raise ValueError(
                f"Expected image size "
                f"{self.image_size}, "
                f"received {(H, W)}."
            )

        if intrins.shape != (
            B,
            N,
            3,
            3,
        ):

            raise ValueError(
                "intrins must have shape "
                f"[{B},{N},3,3]. "
                f"Received {tuple(intrins.shape)}"
            )

        if rots.shape != (
            B,
            N,
            3,
            3,
        ):

            raise ValueError(
                "rots must have shape "
                f"[{B},{N},3,3]."
            )

        if trans.shape != (
            B,
            N,
            3,
        ):

            raise ValueError(
                "trans must have shape "
                f"[{B},{N},3]."
            )

        if post_rots.shape != (
            B,
            N,
            3,
            3,
        ):

            raise ValueError(
                "post_rots must have shape "
                f"[{B},{N},3,3]."
            )

        if post_trans.shape != (
            B,
            N,
            3,
        ):

            raise ValueError(
                "post_trans must have shape "
                f"[{B},{N},3]."
            )

    # ==================================================================
    # Forward
    # ==================================================================

    def forward(
        self,
        images: torch.Tensor,
        intrins: torch.Tensor,
        rots: torch.Tensor,
        trans: torch.Tensor,
        post_rots: torch.Tensor,
        post_trans: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        self._check_inputs(
            images,
            intrins,
            rots,
            trans,
            post_rots,
            post_trans,
        )

        # ==========================================================
        # STEP 1
        # Camera Encoder
        # ==========================================================

        camera_features = self.camera_encoder(
            images
        )

        # Expected:
        #
        # [B,N,512,8,22]

        # Safety check.
        if camera_features.shape[-2:] != self.feature_size:

            raise RuntimeError(
                "CameraEncoder produced unexpected "
                "spatial resolution.\n"
                f"Expected {self.feature_size}, "
                f"received "
                f"{tuple(camera_features.shape[-2:])}."
            )

        # ==========================================================
        # STEP 2
        # Lift
        # ==========================================================

        lift_output = self.lift(
            camera_features
        )

        depth_logits = lift_output[
            "depth_logits"
        ]

        depth_probs = lift_output[
            "depth_probs"
        ]

        context = lift_output[
            "context"
        ]

        lifted_features = lift_output[
            "lifted_features"
        ]

        # lifted_features:
        #
        # [B,N,D,Hf,Wf,C]

        # ==========================================================
        # STEP 3
        # Geometry
        # ==========================================================

        geometry = self.geometry(
            intrins=intrins,
            rots=rots,
            trans=trans,
            post_rots=post_rots,
            post_trans=post_trans,
        )

        # geometry:
        #
        # [B,N,D,Hf,Wf,3]

        # ==========================================================
        # Critical alignment check
        # ==========================================================

        if (
            lifted_features.shape[:5]
            != geometry.shape[:5]
        ):

            raise RuntimeError(
                "Lift and Geometry are not aligned.\n"
                f"Lift: {tuple(lifted_features.shape)}\n"
                f"Geometry: {tuple(geometry.shape)}"
            )

        # ==========================================================
        # STEP 4
        # SPLAT / Voxel Pooling
        # ==========================================================

        bev_raw = self.voxel_pooling(
            features=lifted_features,
            geometry=geometry,
        )

        # Expected:
        #
        # [B,64,200,200]

        # ==========================================================
        # STEP 5
        # BEV Backbone
        # ==========================================================

        bev_features = self.bev_backbone(
            bev_raw
        )

        # Expected:
        #
        # [B,128,200,200]

        # ==========================================================
        # STEP 6
        # Detection Head
        # ==========================================================

        predictions = self.detection_head(
            bev_features
        )

        # ==========================================================
        # Normal mode
        # ==========================================================

        if not self.return_intermediates:

            return predictions

        # ==========================================================
        # Debug / pedagogical mode
        # ==========================================================
        #
        # Très utile pendant notre développement.
        #
        # On peut inspecter chaque étage du modèle.
        #

        return {
            "predictions": predictions,

            "camera_features": camera_features,

            "depth_logits": depth_logits,

            "depth_probs": depth_probs,

            "context": context,

            "lifted_features": lifted_features,

            "geometry": geometry,

            "bev_raw": bev_raw,

            "bev_features": bev_features,
        }