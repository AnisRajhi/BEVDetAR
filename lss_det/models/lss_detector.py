#!/usr/bin/env python3
"""
LSSDetector v2

    images ─► CameraEncoder ─► CameraAwareLift ─┐ (depth_logits -> loss profondeur lidar)
                                ▲               │
       intrins/rots/trans/post ─┤               ▼
                                └─► Geometry(+bda) ─► VoxelPooling ─► BEVBackbone ─► DetectionHead

Sorties : heatmap, reg, center_z, dim (log), rot (sin, cos), depth_logits.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from lss_det import config as C
from lss_det.models.bev_backbone import BEVBackbone
from lss_det.models.camera_encoder import CameraEncoder
from lss_det.models.detection_head import DetectionHead
from lss_det.models.geometry import Geometry
from lss_det.models.lift import CameraAwareLift, camera_parameters
from lss_det.models.voxel_pooling import VoxelPooling


class LSSDetector(nn.Module):
    def __init__(
        self,
        num_classes: int = len(C.CLASSES),
        camera_pretrained: bool = C.CAMERA_PRETRAINED,
        camera_weights_path: Optional[str] = None,
        image_size=C.IMAGE_SIZE,
        downsample: int = C.DOWNSAMPLE,
        depth_bound=C.DEPTH_BOUND,
        xbound=C.XBOUND,
        ybound=C.YBOUND,
        zbound=C.ZBOUND,
        return_intermediates: bool = False,
    ):
        super().__init__()
        self.image_size = tuple(image_size)
        self.return_intermediates = bool(return_intermediates)

        self.camera_encoder = CameraEncoder(
            out_channels=C.CAMERA_FEATURE_CHANNELS,
            pretrained=camera_pretrained,
            weights_path=camera_weights_path,
            freeze_blocks=C.FREEZE_BACKBONE_BLOCKS,
        )
        self.geometry = Geometry(self.image_size, downsample, depth_bound)
        self.lift = CameraAwareLift(
            in_channels=C.CAMERA_FEATURE_CHANNELS,
            mid_channels=C.DEPTHNET_MID_CHANNELS,
            num_depth_bins=self.geometry.num_depth_bins,
            context_channels=C.CONTEXT_CHANNELS,
        )
        self.voxel_pooling = VoxelPooling(xbound=xbound, ybound=ybound, zbound=zbound, collapse_z=True)
        nz = self.voxel_pooling.Nz
        self.bev_backbone = BEVBackbone(in_channels=C.CONTEXT_CHANNELS * nz, out_channels=C.BEV_CHANNELS)
        self.detection_head = DetectionHead(
            in_channels=C.BEV_CHANNELS, shared_channels=C.BEV_CHANNELS, head_channels=64, num_classes=num_classes,
        )

    def forward(self, images, intrins, rots, trans, post_rots, post_trans, bda=None) -> Dict[str, torch.Tensor]:
        B, N = images.shape[:2]
        if tuple(images.shape[-2:]) != self.image_size:
            raise ValueError(f"Image {tuple(images.shape[-2:])} != {self.image_size}")
        if bda is None:
            bda = torch.eye(3, device=images.device, dtype=intrins.dtype).expand(B, 3, 3)

        feats = self.camera_encoder(images)
        cam_params = camera_parameters(intrins, rots, trans, post_rots, post_trans, self.image_size)
        lift = self.lift(feats, cam_params)
        geom = self.geometry(intrins, rots, trans, post_rots, post_trans, bda)

        bev_raw = self.voxel_pooling(lift["lifted_features"], geom)
        bev = self.bev_backbone(bev_raw)
        out = self.detection_head(bev)
        out["depth_logits"] = lift["depth_logits"]

        if self.return_intermediates:
            out.update({
                "camera_features": feats,
                "depth_probs": lift["depth_probs"],
                "context": lift["context"],
                "geometry": geom,
                "bev_raw": bev_raw,
                "bev_features": bev,
            })
        return out

    def parameter_groups(self, lr: float, backbone_lr_mult: float, weight_decay: float):
        """
        3 groupes : backbone pré-entraîné (LR réduit), reste avec weight
        decay, et normalisations/biais sans weight decay.
        """
        backbone_ids = {id(p) for p in self.camera_encoder.backbone_parameters()}
        backbone, decay, no_decay = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in backbone_ids:
                backbone.append(p)
            elif p.ndim <= 1:
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": backbone, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay, "name": "backbone"},
            {"params": decay, "lr": lr, "weight_decay": weight_decay, "name": "decay"},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0, "name": "no_decay"},
        ]
