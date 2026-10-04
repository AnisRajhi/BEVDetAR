#!/usr/bin/env python3
"""
Garde-fou contre le piège "ReLU morte" de la tête heatmap.

Si l'information de l'objet est présente dans le BEV, BEVBackbone +
DetectionHead doivent faire monter la probabilité au centre GT. Avec
l'ancienne init (std=0.001), elle restait figée à sigmoid(-2.19) = 0.10.
"""
import torch

from lss_det.engine import gt_center_probability
from lss_det.losses.detection_loss import gaussian_focal_loss
from lss_det.models.bev_backbone import BEVBackbone
from lss_det.models.detection_head import DetectionHead
from lss_det.targets.centerpoint_targets import CenterPointTargetBuilder

from lss_det import config as C

# Grille RÉELLE (128x128) : sur une grille réduite, il y a moins de
# cellules de fond et le piège ne se referme pas (test non discriminant).
XB, YB = C.XBOUND, C.YBOUND


def test_heatmap_init_is_not_tiny():
    head = DetectionHead(128, 128, 64, 6)
    assert head.heatmap_head.net[-1].weight.std().item() > 0.01
    assert torch.allclose(head.heatmap_head.net[-1].bias, torch.full((6,), -2.19))


def test_heatmap_learns_when_bev_contains_the_objects():
    torch.manual_seed(0)
    gt = torch.tensor([[12.0, 3.0, 0.9, 4.5, 1.9, 1.6, 0.3], [9.0, -2.0, 0.9, 0.7, 0.7, 1.8, 1.0]])
    lab = torch.tensor([0, 3])
    tg = CenterPointTargetBuilder(xbound=XB, ybound=YB)([gt], [lab])
    n = tg["heatmap"].shape[-1]
    bev = 0.3 * torch.randn(1, 64, n, n)
    for (x, y, *_), k in zip(gt.tolist(), lab.tolist()):
        ix, iy = int((x - XB[0]) / XB[2]), int((y - YB[0]) / YB[2])
        bev[0, 8 * k:8 * k + 8, ix - 2:ix + 3, iy - 2:iy + 3] += 2.0

    bb, head = BEVBackbone(64, 128), DetectionHead(128, 128, 64, 6)
    opt = torch.optim.Adam(list(bb.parameters()) + list(head.parameters()), lr=2e-4)
    for _ in range(50):
        out = head(bb(bev))
        loss = gaussian_focal_loss(out["heatmap"], tg["heatmap"])
        opt.zero_grad()
        loss.backward()
        opt.step()
    # mesuré : init par défaut ~0.3 ; ancienne init std=0.001 : 0.10 (figé)
    assert gt_center_probability(head(bb(bev)), tg) > 0.18
