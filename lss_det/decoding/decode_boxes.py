#!/usr/bin/env python3
"""
Décodage CenterPoint : heatmap -> maxima locaux -> top-K -> boîtes
-> seuil -> circle-NMS par classe (rayon dépendant de la classe).

v2 : dimensions = exp(log-dim) ; rayon NMS par classe (un rayon unique
de 1 m supprimait mal les doublons de bus/camions et pouvait fusionner
des piétons voisins).
"""

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from lss_det import config as C
from lss_det.losses.detection_loss import gather_feature_map


def local_maximum_suppression(heatmap: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = (kernel_size - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel_size, stride=1, padding=pad)
    return heatmap * (heatmap == hmax).to(heatmap.dtype)


def circle_nms(centers: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, radius_per_label: torch.Tensor):
    keep = []
    for c in torch.unique(labels):
        idx = torch.nonzero(labels == c).squeeze(1)
        idx = idx[torch.argsort(scores[idx], descending=True)]
        r = float(radius_per_label[int(c)])
        while idx.numel() > 0:
            cur = idx[0]
            keep.append(cur)
            if idx.numel() == 1:
                break
            rest = idx[1:]
            dist = torch.linalg.norm(centers[rest] - centers[cur], dim=1)
            idx = rest[dist >= r]
    if not keep:
        return torch.zeros(0, dtype=torch.long, device=centers.device)
    keep = torch.stack(keep)
    return keep[torch.argsort(scores[keep], descending=True)]


class CenterPointDecoder(nn.Module):
    def __init__(
        self,
        classes: Sequence[str] = C.CLASSES,
        xbound=C.XBOUND,
        ybound=C.YBOUND,
        top_k: int = C.DECODER_TOP_K,
        score_threshold: float = C.DECODER_SCORE_THRESHOLD,
        nms_radius: Dict[str, float] = C.NMS_RADIUS,
    ):
        super().__init__()
        self.x_min, self.dx = float(xbound[0]), float(xbound[2])
        self.y_min, self.dy = float(ybound[0]), float(ybound[2])
        self.top_k = int(top_k)
        self.score_threshold = float(score_threshold)
        self.register_buffer(
            "nms_radius", torch.tensor([nms_radius[c] for c in classes], dtype=torch.float32), persistent=False
        )

    @torch.no_grad()
    def forward(self, predictions: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        heat = torch.sigmoid(predictions["heatmap"].float())
        B, K, Nx, Ny = heat.shape
        heat = local_maximum_suppression(heat)
        scores, top = torch.topk(heat.reshape(B, -1), k=min(self.top_k, K * Nx * Ny), dim=1)
        labels = top // (Nx * Ny)
        spatial = top % (Nx * Ny)
        ix, iy = spatial // Ny, spatial % Ny

        reg = gather_feature_map(predictions["reg"].float(), spatial)
        z = gather_feature_map(predictions["center_z"].float(), spatial)[..., 0]
        dims = gather_feature_map(predictions["dim"].float(), spatial).clamp(-5.0, 5.0).exp()
        rot = gather_feature_map(predictions["rot"].float(), spatial)

        x = self.x_min + (ix.float() + reg[..., 0]) * self.dx
        y = self.y_min + (iy.float() + reg[..., 1]) * self.dy
        yaw = torch.atan2(rot[..., 0], rot[..., 1])
        boxes = torch.stack([x, y, z, dims[..., 0], dims[..., 1], dims[..., 2], yaw], dim=-1)

        results = []
        for b in range(B):
            valid = scores[b] >= self.score_threshold
            bx, sc, lb = boxes[b][valid], scores[b][valid], labels[b][valid]
            if bx.shape[0] > 0:
                keep = circle_nms(bx[:, :2], sc, lb, self.nms_radius)
                bx, sc, lb = bx[keep], sc[keep], lb[keep]
            results.append({"boxes": bx, "scores": sc, "labels": lb})
        return results
