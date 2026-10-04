#!/usr/bin/env python3
"""
Geometry LSS : frustum (u, v, d) -> points 3D dans le repère BEV.

Corrections par rapport à la v1
-------------------------------
1. Position des rayons.
   v1 : u = linspace(0, W-1, Wf)  ->  u_j = j * (W-1)/(Wf-1)
   Une cellule de feature (stride 16) correspond pourtant au bloc de
   pixels [16 j, 16 j + 15], centré en 16 j + 7.5. L'écart allait de
   -7.5 px (bord gauche) à +7.5 px (bord droit) : ~1.5° de biais
   angulaire, soit ~1 m latéral à 40 m, systématique par colonne et
   impossible à corriger par le réseau (la géométrie est figée).
   Ici : u_j = 16 j + 7.5, v_i = 16 i + 7.5. C'est aussi exactement la
   convention des labels de profondeur lidar.

2. Profondeur au CENTRE du bin (d_min + (k + 0.5) * pas), cohérent avec
   le label de bin floor((d - d_min) / pas).

3. BEV Data Augmentation : P_bev = bda @ P_ego.

`image_to_ego()` est exposé pour pouvoir tester la chaîne géométrique
complète (projection lidar <-> déprojection frustum) point par point.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class Geometry(nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int],
        downsample: int,
        depth_bound: Tuple[float, float, float],
    ):
        super().__init__()
        self.image_h, self.image_w = int(image_size[0]), int(image_size[1])
        self.downsample = int(downsample)
        self.feature_h = self.image_h // self.downsample
        self.feature_w = self.image_w // self.downsample
        self.depth_min, self.depth_max, self.depth_step = (float(v) for v in depth_bound)
        self.num_depth_bins = int(round((self.depth_max - self.depth_min) / self.depth_step))
        self.register_buffer("frustum", self.create_frustum(), persistent=False)

    def create_frustum(self) -> torch.Tensor:
        D, Hf, Wf, s = self.num_depth_bins, self.feature_h, self.feature_w, self.downsample
        ds = self.depth_min + (torch.arange(D, dtype=torch.float32) + 0.5) * self.depth_step
        xs = torch.arange(Wf, dtype=torch.float32) * s + (s - 1) / 2.0
        ys = torch.arange(Hf, dtype=torch.float32) * s + (s - 1) / 2.0
        d, v, u = torch.meshgrid(ds, ys, xs, indexing="ij")
        return torch.stack([u, v, d], dim=-1)        # [D, Hf, Wf, 3]

    @staticmethod
    def image_to_ego(
        uvd: torch.Tensor,          # [B, N, P, 3]  (u', v', profondeur) image AUGMENTÉE
        intrins: torch.Tensor,      # [B, N, 3, 3]
        rots: torch.Tensor,         # [B, N, 3, 3]  caméra -> ego
        trans: torch.Tensor,        # [B, N, 3]
        post_rots: torch.Tensor,    # [B, N, 3, 3]
        post_trans: torch.Tensor,   # [B, N, 3]
        bda: Optional[torch.Tensor] = None,   # [B, 3, 3]
    ) -> torch.Tensor:
        # 1. annuler l'augmentation image
        pts = uvd - post_trans[:, :, None, :]
        pts = torch.einsum("bnij,bnpj->bnpi", torch.linalg.inv(post_rots), pts)
        # 2. (u, v, d) -> (u d, v d, d)
        pts = torch.cat([pts[..., :2] * pts[..., 2:3], pts[..., 2:3]], dim=-1)
        # 3. caméra -> ego
        combine = rots @ torch.linalg.inv(intrins)
        pts = torch.einsum("bnij,bnpj->bnpi", combine, pts) + trans[:, :, None, :]
        # 4. ego -> BEV augmenté
        if bda is not None:
            pts = torch.einsum("bij,bnpj->bnpi", bda, pts)
        return pts

    def forward(self, intrins, rots, trans, post_rots, post_trans, bda=None) -> torch.Tensor:
        B, N = intrins.shape[:2]
        D, Hf, Wf = self.frustum.shape[:3]
        uvd = self.frustum.to(intrins.dtype).reshape(1, 1, -1, 3).expand(B, N, -1, 3)
        pts = self.image_to_ego(uvd, intrins, rots, trans, post_rots, post_trans, bda)
        return pts.reshape(B, N, D, Hf, Wf, 3)
