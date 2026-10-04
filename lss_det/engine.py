#!/usr/bin/env python3
"""
Boucles d'entraînement et d'évaluation partagées par train.py et evaluate.py.
"""

import math
import random
from typing import Dict

import numpy as np
import torch

from lss_det import config as C
from lss_det.data.collate import model_inputs
from lss_det.losses.detection_loss import depth_metrics
from lss_det.metrics.detection_metrics import DetectionEvaluator

LOSS_KEYS = ("loss", "loss_heatmap", "loss_bbox", "loss_reg", "loss_z", "loss_dim", "loss_rot", "loss_depth")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_snapshot() -> Dict:
    return {k: v for k, v in vars(C).items() if k.isupper()}


class Meter:
    def __init__(self):
        self.sums, self.weights = {}, {}

    def add(self, values: Dict, weight: float = 1.0):
        for k, v in values.items():
            v = float(v.detach().item()) if torch.is_tensor(v) else float(v)
            if math.isnan(v):
                continue
            self.sums[k] = self.sums.get(k, 0.0) + v * weight
            self.weights[k] = self.weights.get(k, 0.0) + weight

    def mean(self) -> Dict[str, float]:
        return {k: self.sums[k] / max(self.weights[k], 1e-12) for k in self.sums}


@torch.no_grad()
def gt_center_probability(predictions, targets) -> float:
    """Probabilité moyenne de la heatmap aux centres GT (diagnostic, pas une métrique)."""
    prob = torch.sigmoid(predictions["heatmap"].float())
    B, K, Nx, Ny = prob.shape
    m = targets["mask"].to(prob.device)
    if m.sum() == 0:
        return float("nan")
    flat = prob.reshape(B, K, Nx * Ny)
    lab = targets["labels"].to(prob.device)
    idx = targets["indices"].to(prob.device)
    vals = flat[torch.arange(B, device=prob.device)[:, None].expand_as(idx), lab, idx]
    return vals[m].mean().item()


def make_scheduler(optimizer, total_steps: int, warmup_steps: int = C.WARMUP_STEPS, min_ratio: float = C.MIN_LR_RATIO):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _targets_to_device(targets, device):
    return {k: v.to(device, non_blocking=True) for k, v in targets.items()}


def train_one_epoch(model, loader, target_builder, criterion, optimizer, scheduler, device,
                    accum_steps: int = C.ACCUM_STEPS, grad_clip: float = C.GRAD_CLIP) -> Dict[str, float]:
    model.train()
    meter = Meter()
    optimizer.zero_grad(set_to_none=True)
    n = len(loader)

    for it, batch in enumerate(loader):
        targets = _targets_to_device(target_builder(batch["gt_boxes"], batch["gt_labels"]), device)
        depth_bins = batch["depth_bins"].to(device, non_blocking=True)
        preds = model(**model_inputs(batch, device))
        losses = criterion(preds, targets, depth_bins)

        if not torch.isfinite(losses["loss"]):
            raise RuntimeError(f"Loss non finie : { {k: float(v) for k, v in losses.items()} }")

        (losses["loss"] / accum_steps).backward()

        if (it + 1) % accum_steps == 0 or (it + 1) == n:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        dm = depth_metrics(preds["depth_logits"], depth_bins)
        meter.add({k: losses[k] for k in LOSS_KEYS})
        meter.add({"gt_center_prob": gt_center_probability(preds, targets)})
        meter.add({"depth_acc1": dm["depth_acc1"], "depth_absrel": dm["depth_absrel"]}, weight=max(dm["depth_cells"], 1))

    out = meter.mean()
    out["lr"] = optimizer.param_groups[1]["lr"]
    return out


@torch.no_grad()
def evaluate(model, loader, target_builder, criterion, decoder, device) -> Dict:
    model.eval()
    meter = Meter()
    evaluator = DetectionEvaluator()

    for batch in loader:
        targets = _targets_to_device(target_builder(batch["gt_boxes"], batch["gt_labels"]), device)
        depth_bins = batch["depth_bins"].to(device, non_blocking=True)
        preds = model(**model_inputs(batch, device))
        losses = criterion(preds, targets, depth_bins)
        dm = depth_metrics(preds["depth_logits"], depth_bins)

        meter.add({k: losses[k] for k in LOSS_KEYS})
        meter.add({"gt_center_prob": gt_center_probability(preds, targets)})
        meter.add({"depth_acc1": dm["depth_acc1"], "depth_absrel": dm["depth_absrel"]}, weight=max(dm["depth_cells"], 1))

        dets = decoder(preds)
        for b, det in enumerate(dets):
            evaluator.add(batch["sample_token"][b], det["boxes"], det["scores"], det["labels"],
                          batch["gt_boxes"][b], batch["gt_labels"][b])

    return {"losses": meter.mean(), "metrics": evaluator.compute()}


def format_losses(prefix: str, d: Dict[str, float]) -> str:
    keys = [("loss", "total"), ("loss_heatmap", "hm"), ("loss_bbox", "bbox"), ("loss_reg", "reg"),
            ("loss_z", "z"), ("loss_dim", "dim"), ("loss_rot", "rot"), ("loss_depth", "depth"),
            ("depth_acc1", "dAcc"), ("depth_absrel", "dRel"), ("gt_center_prob", "GTp")]
    return f"  {prefix:<5}| " + " | ".join(f"{short}={d[k]:.4f}" for k, short in keys if k in d)
