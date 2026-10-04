#!/usr/bin/env python3
"""
Targets CenterPoint.

Changement v2 : les dimensions sont encodées en log(l), log(w), log(h).
En L1 brut (mètres), un bus de 11 m pesait ~15x plus qu'un piéton dans
la loss de dimension, et la sortie pouvait devenir négative. En log,
l'erreur est relative (10 % d'erreur coûte pareil pour toutes les
classes) et exp() garantit des tailles positives au décodage.
"""

import math
from typing import Dict, List, Sequence, Tuple

import torch

from lss_det import config as C


def gaussian2d(radius: int, device=None, dtype=torch.float32) -> torch.Tensor:
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    return torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))


def gaussian_radius(length_cells: float, width_cells: float, min_overlap: float = 0.1) -> float:
    h, w = float(length_cells), float(width_cells)
    b1 = h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(max(0.0, b1 * b1 - 4 * c1))) / 2
    a2, b2, c2 = 4.0, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 + math.sqrt(max(0.0, b2 * b2 - 4 * a2 * c2))) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(max(0.0, b3 * b3 - 4 * a3 * c3))) / 2
    return min(r1, r2, r3)


def draw_gaussian(heatmap: torch.Tensor, cx: int, cy: int, radius: int) -> None:
    """heatmap [Nx, Ny], indexée heatmap[ix, iy]."""
    Nx, Ny = heatmap.shape
    if not (0 <= cx < Nx and 0 <= cy < Ny):
        return
    g = gaussian2d(radius, heatmap.device, heatmap.dtype)
    l, r = min(cx, radius), min(Nx - cx - 1, radius)
    t, b = min(cy, radius), min(Ny - cy - 1, radius)
    patch = heatmap[cx - l:cx + r + 1, cy - t:cy + b + 1]
    torch.maximum(patch, g[radius - l:radius + r + 1, radius - t:radius + b + 1], out=patch)


class CenterPointTargetBuilder:
    def __init__(
        self,
        classes: Sequence[str] = C.CLASSES,
        xbound: Tuple[float, float, float] = C.XBOUND,
        ybound: Tuple[float, float, float] = C.YBOUND,
        max_objects: int = C.MAX_OBJECTS,
        gaussian_overlap: float = C.GAUSSIAN_OVERLAP,
        min_radius: int = C.MIN_RADIUS,
    ):
        self.num_classes = len(classes)
        self.x_min, self.dx = float(xbound[0]), float(xbound[2])
        self.y_min, self.dy = float(ybound[0]), float(ybound[2])
        self.Nx = int(round((xbound[1] - xbound[0]) / xbound[2]))
        self.Ny = int(round((ybound[1] - ybound[0]) / ybound[2]))
        self.max_objects = int(max_objects)
        self.gaussian_overlap = float(gaussian_overlap)
        self.min_radius = int(min_radius)

    def __call__(self, gt_boxes: List[torch.Tensor], gt_labels: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = len(gt_boxes)
        M = self.max_objects
        heatmap = torch.zeros(B, self.num_classes, self.Nx, self.Ny)
        indices = torch.zeros(B, M, dtype=torch.long)
        mask = torch.zeros(B, M, dtype=torch.bool)
        # [dx, dy, z, log_l, log_w, log_h, sin, cos]
        box_targets = torch.zeros(B, M, 8)
        labels = torch.zeros(B, M, dtype=torch.long)

        for b in range(B):
            boxes, labs = gt_boxes[b].detach().cpu().float(), gt_labels[b].detach().cpu()
            k = 0
            for i in range(boxes.shape[0]):
                if k >= M:
                    break
                x, y, z, l, w, h, yaw = boxes[i].tolist()
                label = int(labs[i])
                if l <= 0 or w <= 0 or h <= 0:
                    continue
                gx, gy = (x - self.x_min) / self.dx, (y - self.y_min) / self.dy
                ix, iy = int(math.floor(gx)), int(math.floor(gy))
                if not (0 <= ix < self.Nx and 0 <= iy < self.Ny):
                    continue

                radius = gaussian_radius(l / self.dx, w / self.dy, self.gaussian_overlap)
                draw_gaussian(heatmap[b, label], ix, iy, max(self.min_radius, int(radius)))

                indices[b, k] = ix * self.Ny + iy
                box_targets[b, k] = torch.tensor(
                    [gx - ix, gy - iy, z, math.log(l), math.log(w), math.log(h), math.sin(yaw), math.cos(yaw)]
                )
                labels[b, k] = label
                mask[b, k] = True
                k += 1

        return {"heatmap": heatmap, "indices": indices, "mask": mask, "box_targets": box_targets, "labels": labels}
