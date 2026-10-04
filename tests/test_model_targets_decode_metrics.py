#!/usr/bin/env python3
import math

import torch

from lss_det import config as C
from lss_det.decoding.decode_boxes import CenterPointDecoder
from lss_det.losses.detection_loss import DetectionLoss, depth_metrics
from lss_det.metrics.detection_metrics import DetectionEvaluator
from lss_det.models.lss_detector import LSSDetector
from lss_det.targets.centerpoint_targets import CenterPointTargetBuilder

GT = torch.tensor([
    [10.3, 2.1, 0.9, 4.5, 1.9, 1.6, 0.3],
    [-20.7, 8.4, 1.0, 11.0, 2.9, 3.4, -2.5],
    [5.2, -14.6, 0.8, 0.7, 0.6, 1.8, 1.2],
])
LABELS = torch.tensor([0, 2, 3])


def perfect_predictions(targets):
    B, K, Nx, Ny = targets["heatmap"].shape
    logits = torch.logit(targets["heatmap"].clamp(1e-4, 1 - 1e-4))
    box = torch.zeros(B, 8, Nx * Ny)
    for b in range(B):
        m = targets["mask"][b]
        box[b][:, targets["indices"][b][m]] = targets["box_targets"][b][m].T
    box = box.reshape(B, 8, Nx, Ny)
    return {"heatmap": logits, "reg": box[:, 0:2], "center_z": box[:, 2:3], "dim": box[:, 3:6], "rot": box[:, 6:8]}


def test_targets_decode_roundtrip_and_metrics():
    targets = CenterPointTargetBuilder()([GT], [LABELS])
    preds = perfect_predictions(targets)
    det = CenterPointDecoder()(preds)[0]
    assert det["boxes"].shape[0] == 3
    order = torch.argsort(det["boxes"][:, 0])
    ref = torch.argsort(GT[:, 0])
    assert torch.allclose(det["boxes"][order], GT[ref], atol=1e-3)

    ev = DetectionEvaluator()
    ev.add("s0", det["boxes"], det["scores"], det["labels"], GT, LABELS)
    res = ev.compute()
    for name in ("car", "bus", "pedestrian"):
        assert abs(res["per_class"][name]["mAP"] - 1.0) < 1e-6
    assert res["ATE"] < 1e-3 and res["ASE"] < 1e-3 and res["AOE"] < 1e-3


def test_metrics_distance_thresholds():
    ev = DetectionEvaluator(classes=["car"], class_range={"car": 50.0})
    pred = GT[:1].clone()
    pred[0, 0] += 1.5
    ev.add("s0", pred, torch.tensor([0.9]), torch.tensor([0]), GT[:1], torch.tensor([0]))
    res = ev.compute()
    ap = res["per_class"]["car"]["AP"]
    assert ap[0.5] == 0.0 and ap[1.0] == 0.0
    assert abs(ap[2.0] - 1.0) < 1e-6 and abs(ap[4.0] - 1.0) < 1e-6
    assert abs(res["mAP"] - 0.5) < 1e-6
    assert abs(res["per_class"]["car"]["ATE"] - 1.5) < 1e-4


def test_model_forward_backward_small():
    torch.manual_seed(0)
    m = LSSDetector(camera_pretrained=False, image_size=(128, 352))
    B, N = 1, 6
    imgs = torch.randn(B, N, 3, 128, 352)
    K = torch.tensor([[1266.0, 0, 816], [0, 1266.0, 491], [0, 0, 1]]).repeat(B, N, 1, 1)
    R = torch.tensor([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]]).repeat(B, N, 1, 1)
    t = torch.tensor([1.5, 0.0, 1.5]).repeat(B, N, 1)
    pr = torch.eye(3).repeat(B, N, 1, 1)
    pr[..., 0, 0] = pr[..., 1, 1] = 0.22
    pt = torch.tensor([0.0, -70.0, 0.0]).repeat(B, N, 1)
    out = m(imgs, K, R, t, pr, pt, torch.eye(3)[None])
    assert out["heatmap"].shape == (B, len(C.CLASSES), C.NX, C.NY)
    assert out["depth_logits"].shape == (B, N, C.NUM_DEPTH_BINS, 8, 22)

    targets = CenterPointTargetBuilder()([GT], [LABELS])
    depth_bins = torch.randint(-1, C.NUM_DEPTH_BINS, (B, N, 8, 22))
    losses = DetectionLoss()(out, targets, depth_bins)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert m.lift.depth_out.weight.grad.abs().sum() > 0
    assert m.camera_encoder.fusion.net[0].weight.grad.abs().sum() > 0
    dm = depth_metrics(out["depth_logits"], depth_bins)
    assert 0.0 <= dm["depth_acc1"] <= 1.0

    groups = m.parameter_groups(2e-4, 0.1, 1e-2)
    assert abs(groups[0]["lr"] - 2e-5) < 1e-12 and len(groups[0]["params"]) > 0
    assert sum(len(g["params"]) for g in groups) == sum(1 for p in m.parameters() if p.requires_grad)
