#!/usr/bin/env python3
"""
Loss totale = heatmap (focal gaussienne)
            + BBOX_WEIGHT  * somme pondérée des L1 [dx, dy, z, log l, log w, log h, sin, cos]
            + DEPTH_WEIGHT * BCE profondeur (labels lidar, formulation BEVDepth)

Changements v2
--------------
- v1 : 4 termes de régression de poids 1 chacun + dimensions en mètres
  -> la régression (surtout dim) pouvait dominer les gradients du tronc
  commun. v2 : pondération CenterPoint (0.25) + log-dimensions.
- Nouveau terme de profondeur : dense, géométrique, transférable.
- Tout est calculé en float32 (robuste si on active l'AMP).
"""

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from lss_det import config as C


def gather_feature_map(prediction: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """prediction [B,C,Nx,Ny], indices [B,M] (ix*Ny+iy) -> [B,M,C]."""
    B, Cc, Nx, Ny = prediction.shape
    flat = prediction.permute(0, 2, 3, 1).reshape(B, Nx * Ny, Cc)
    return torch.gather(flat, 1, indices.unsqueeze(-1).expand(-1, -1, Cc))


def gaussian_focal_loss(logits, target, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-4):
    pred = torch.sigmoid(logits.float()).clamp(eps, 1.0 - eps)
    target = target.float()
    pos = target.eq(1.0).float()
    neg = 1.0 - pos
    pos_loss = torch.log(pred) * (1 - pred) ** alpha * pos
    neg_loss = torch.log(1 - pred) * pred ** alpha * (1 - target) ** beta * neg
    num_pos = pos.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def depth_loss_bevdepth(depth_logits: torch.Tensor, depth_bins: torch.Tensor) -> torch.Tensor:
    """
    depth_logits [B,N,D,Hf,Wf] ; depth_bins [B,N,Hf,Wf] (-1 = pas de label).
    BCE entre la distribution softmax et le one-hot, sommée sur D et
    moyennée sur les cellules labellisées (BEVDepth).
    """
    D = depth_logits.shape[2]
    probs = depth_logits.float().softmax(dim=2).permute(0, 1, 3, 4, 2).reshape(-1, D)
    labels = depth_bins.reshape(-1)
    fg = labels >= 0
    if fg.sum() == 0:
        return depth_logits.sum() * 0.0
    onehot = F.one_hot(labels[fg], D).float()
    bce = F.binary_cross_entropy(probs[fg].clamp(1e-6, 1 - 1e-6), onehot, reduction="sum")
    return bce / fg.sum()


@torch.no_grad()
def depth_metrics(depth_logits: torch.Tensor, depth_bins: torch.Tensor, depth_bound=C.DEPTH_BOUND) -> Dict[str, float]:
    """
    acc1   : fraction des cellules où argmax est à ±1 bin du label
    absrel : |E[d] - d_gt| / d_gt  (E[d] = espérance de la distribution)
    """
    dmin, _, step = depth_bound
    D = depth_logits.shape[2]
    probs = depth_logits.float().softmax(dim=2)
    labels = depth_bins
    fg = labels >= 0
    if fg.sum() == 0:
        return {"depth_acc1": float("nan"), "depth_absrel": float("nan"), "depth_cells": 0}
    centers = dmin + (torch.arange(D, device=probs.device, dtype=probs.dtype) + 0.5) * step
    expected = (probs * centers.view(1, 1, D, 1, 1)).sum(dim=2)
    argmax = probs.argmax(dim=2)
    gt_depth = dmin + (labels.clamp(min=0).float() + 0.5) * step
    acc1 = ((argmax - labels).abs() <= 1)[fg].float().mean().item()
    absrel = ((expected - gt_depth).abs() / gt_depth)[fg].mean().item()
    return {"depth_acc1": acc1, "depth_absrel": absrel, "depth_cells": int(fg.sum().item())}


class DetectionLoss(nn.Module):
    COMPONENTS = ("reg", "z", "dim", "rot")
    SLICES = {"reg": slice(0, 2), "z": slice(2, 3), "dim": slice(3, 6), "rot": slice(6, 8)}

    def __init__(
        self,
        heatmap_weight: float = C.HEATMAP_WEIGHT,
        bbox_weight: float = C.BBOX_WEIGHT,
        code_weights: Sequence[float] = C.BBOX_CODE_WEIGHTS,
        depth_weight: float = C.DEPTH_WEIGHT,
    ):
        super().__init__()
        self.heatmap_weight = float(heatmap_weight)
        self.bbox_weight = float(bbox_weight)
        self.depth_weight = float(depth_weight)
        self.register_buffer("code_weights", torch.tensor(code_weights, dtype=torch.float32), persistent=False)

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], depth_bins=None):
        hm = gaussian_focal_loss(predictions["heatmap"], targets["heatmap"])

        box_pred = torch.cat(
            [predictions["reg"], predictions["center_z"], predictions["dim"], predictions["rot"]], dim=1
        ).float()
        box_pred = gather_feature_map(box_pred, targets["indices"])           # [B,M,8]
        mask = targets["mask"].float().unsqueeze(-1)
        num = mask.sum().clamp(min=1.0)
        per_dim = ((box_pred - targets["box_targets"].float()).abs() * mask).sum(dim=(0, 1)) / num   # [8]
        bbox = (per_dim * self.code_weights).sum()

        total = self.heatmap_weight * hm + self.bbox_weight * bbox
        out = {"loss_heatmap": hm, "loss_bbox": bbox}
        for name in self.COMPONENTS:
            out[f"loss_{name}"] = per_dim[self.SLICES[name]].sum().detach()

        if depth_bins is not None and self.depth_weight > 0:
            dl = depth_loss_bevdepth(predictions["depth_logits"], depth_bins)
            total = total + self.depth_weight * dl
            out["loss_depth"] = dl
        else:
            out["loss_depth"] = torch.zeros((), device=hm.device)

        out["loss"] = total
        return out
