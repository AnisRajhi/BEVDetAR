#!/usr/bin/env python3
"""
Lift "camera-aware" (inspiré de BEVDepth).

Problème de la v1
-----------------
La profondeur sortait d'UNE conv 1x1 partagée par les 6 caméras.
Or nuScenes n'a pas des caméras identiques : CAM_BACK a une focale
d'environ 800 px contre ~1260 px pour les autres. Une voiture à 20 m
n'a donc pas la même taille apparente selon la caméra, et nos
augmentations (resize, flip, rotation) modifient encore les
intrinsèques effectives. Inférer une profondeur métrique à partir de
l'apparence SANS connaître la focale est mal posé : le réseau compense
en mémorisant les scènes.

Ici :
  1. on calcule les intrinsèques EFFECTIVES  K_eff = post_rot @ K (+ post_trans)
     et on les concatène aux extrinsèques (sans BDA) -> vecteur caméra ;
  2. un MLP transforme ce vecteur en gains par canal (squeeze-excitation)
     appliqués séparément aux branches profondeur et contexte ;
  3. la branche profondeur a deux blocs résiduels 3x3 (champ réceptif
     suffisant pour "voir" l'objet entier, son contact au sol, etc.).

Le gain vaut exactement 1 à l'initialisation (2*sigmoid(0)), donc le
module démarre comme un réseau non conditionné.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from lss_det.models.bev_backbone import BasicBlock, make_group_norm

CAMERA_PARAM_DIM = 18


def camera_parameters(
    intrins: torch.Tensor,
    rots: torch.Tensor,
    trans: torch.Tensor,
    post_rots: torch.Tensor,
    post_trans: torch.Tensor,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Retourne [B, N, 18] :
        6 coefficients de K_eff (normalisés par W, H) + R (9) + t (3).
    """
    H, W = image_size
    K_eff = post_rots @ intrins
    shift = torch.zeros_like(K_eff)
    shift[..., :2, 2] = post_trans[..., :2]
    K_eff = K_eff + shift

    k = torch.stack(
        [
            K_eff[..., 0, 0] / W, K_eff[..., 0, 1] / W, K_eff[..., 0, 2] / W,
            K_eff[..., 1, 0] / H, K_eff[..., 1, 1] / H, K_eff[..., 1, 2] / H,
        ],
        dim=-1,
    )
    return torch.cat([k, rots.flatten(-2), trans], dim=-1)


class CameraGate(nn.Module):
    def __init__(self, cam_dim: int, channels: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cam_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, cam):
        gate = 2.0 * torch.sigmoid(self.mlp(cam))
        return x * gate[:, :, None, None]


class CameraAwareLift(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        num_depth_bins: int,
        context_channels: int,
        cam_param_dim: int = CAMERA_PARAM_DIM,
    ):
        super().__init__()
        self.D = int(num_depth_bins)
        self.C = int(context_channels)

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            make_group_norm(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.depth_gate = CameraGate(cam_param_dim, mid_channels)
        self.context_gate = CameraGate(cam_param_dim, mid_channels)

        self.depth_blocks = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        )
        self.depth_out = nn.Conv2d(mid_channels, self.D, 1)
        self.context_out = nn.Conv2d(mid_channels, self.C, 1)

        # Distribution uniforme au départ.
        nn.init.zeros_(self.depth_out.weight)
        nn.init.zeros_(self.depth_out.bias)

    def forward(self, image_features: torch.Tensor, cam_params: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, Cin, Hf, Wf = image_features.shape
        x = self.reduce(image_features.reshape(B * N, Cin, Hf, Wf))
        cam = cam_params.reshape(B * N, -1)

        depth_logits = self.depth_out(self.depth_blocks(self.depth_gate(x, cam)))
        context = self.context_out(self.context_gate(x, cam))
        depth_probs = depth_logits.softmax(dim=1)

        # [BN, C, D, Hf, Wf] -> [B, N, D, Hf, Wf, C]
        lifted = depth_probs.unsqueeze(1) * context.unsqueeze(2)
        lifted = lifted.reshape(B, N, self.C, self.D, Hf, Wf).permute(0, 1, 3, 4, 5, 2).contiguous()

        return {
            "depth_logits": depth_logits.reshape(B, N, self.D, Hf, Wf),
            "depth_probs": depth_probs.reshape(B, N, self.D, Hf, Wf),
            "context": context.reshape(B, N, self.C, Hf, Wf),
            "lifted_features": lifted,
        }
