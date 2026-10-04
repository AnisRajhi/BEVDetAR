#!/usr/bin/env python3
"""
CameraEncoder : EfficientNet-B0 (pré-entraîné ImageNet) + fusion r4/r5.

Différences avec la v1
----------------------
- On n'appelle plus `extract_endpoints()`, qui calculait aussi
  `_conv_head` (320 -> 1280 canaux) pour rien. `_conv_head`, `_bn1` et
  `_fc` (1.69 M paramètres jamais utilisés, comptés comme "trainables")
  sont supprimés.
- Fusion réduite à 256 canaux : la v1 avait 4.35 M paramètres ALÉATOIRES
  dans la fusion, soit plus que la partie utile d'EfficientNet (~3.6 M).
- `backbone_parameters()` permet un LR plus faible sur la partie
  pré-entraînée ; `freeze_blocks` permet d'en geler le début.
- BN EfficientNet figées (statistiques + affine), DropConnect désactivé :
  inchangé, c'était la bonne correction du problème train/eval.
"""

from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


def make_group_norm(num_channels: int) -> nn.GroupNorm:
    for g in (32, 16, 8, 4, 2, 1):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    raise RuntimeError(num_channels)


class FeatureFusion(nn.Module):
    """r4 [112, H/16] + upsample(r5 [320, H/32]) -> [out, H/16]."""

    def __init__(self, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(112 + 320, out_channels, 3, padding=1, bias=False),
            make_group_norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            make_group_norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, r4, r5):
        r5 = F.interpolate(r5, size=r4.shape[-2:], mode="bilinear", align_corners=True)
        return self.net(torch.cat([r4, r5], dim=1))


class CameraEncoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 256,
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        freeze_blocks: int = 0,
    ):
        super().__init__()
        self.out_channels = int(out_channels)

        if pretrained:
            kwargs = {"weights_path": weights_path} if weights_path else {}
            self.backbone = EfficientNet.from_pretrained("efficientnet-b0", **kwargs)
        else:
            self.backbone = EfficientNet.from_name("efficientnet-b0")

        for name in ("_conv_head", "_bn1", "_fc", "_avg_pooling", "_dropout"):
            if hasattr(self.backbone, name):
                delattr(self.backbone, name)

        self.fusion = FeatureFusion(self.out_channels)

        self.freeze_blocks = int(freeze_blocks)
        if self.freeze_blocks > 0:
            for p in self.backbone._conv_stem.parameters():
                p.requires_grad_(False)
            for block in self.backbone._blocks[: self.freeze_blocks]:
                for p in block.parameters():
                    p.requires_grad_(False)

        self._freeze_backbone_bn()

    # --------------------------------------------------------------

    def _freeze_backbone_bn(self):
        for m in self.backbone.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_backbone_bn()
        return self

    def backbone_parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self.backbone.parameters() if p.requires_grad)

    # --------------------------------------------------------------

    def _extract_r4_r5(self, x):
        bb = self.backbone
        x = bb._swish(bb._bn0(bb._conv_stem(x)))
        endpoints = []
        prev = x
        for block in bb._blocks:
            x = block(x, drop_connect_rate=0.0)
            if prev.size(2) > x.size(2):
                endpoints.append(prev)
            prev = x
        endpoints.append(x)
        # endpoints : strides 2, 4, 8, 16, 32
        return endpoints[3], endpoints[4]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"CameraEncoder attend [B,N,3,H,W], reçu {tuple(images.shape)}")
        B, N, _, H, W = images.shape
        r4, r5 = self._extract_r4_r5(images.reshape(B * N, 3, H, W))
        feats = self.fusion(r4, r5)
        return feats.reshape(B, N, *feats.shape[1:])
