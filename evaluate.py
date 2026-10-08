#!/usr/bin/env python3
"""
Évalue un checkpoint sur le split de validation.

    python3 evaluate.py

Affiche : mAP (seuils 0.5/1/2/4 m), ATE/ASE/AOE, AP par classe,
rappel par tranche de distance, et qualité de la profondeur prédite
(dAcc = argmax à ±1 bin du lidar, dRel = erreur relative de E[d]).
"""

import sys

import torch
from torch.utils.data import DataLoader

from lss_det import config as C
from lss_det.data.collate import lss_collate_fn
from lss_det.data.nuscenes_dataset import NuScenesLSSDataset, release_nuscenes
from lss_det.decoding.decode_boxes import CenterPointDecoder
from lss_det.engine import evaluate, format_losses
from lss_det.losses.detection_loss import DetectionLoss
from lss_det.metrics.detection_metrics import format_report
from lss_det.models.lss_detector import LSSDetector
from lss_det.targets.centerpoint_targets import CenterPointTargetBuilder

CHECKPOINT = "checkpoints_part010203/reference_v231_part010203.pt"
SPLIT = "val"


def main():
    # Affiche chaque ligne immédiatement, même à travers `| tee` (sinon Python
    # retient ~8 Ko de sortie et l'entraînement semble bloqué).
    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

    saved = ckpt.get("config", {})
    for key in ("IMAGE_SIZE", "DEPTH_BOUND", "XBOUND", "YBOUND", "ZBOUND", "CLASSES"):
        if key in saved and tuple(saved[key]) != tuple(getattr(C, key)):
            raise ValueError(f"{key} du checkpoint ({saved[key]}) != config actuelle ({getattr(C, key)})")

    model = LSSDetector(camera_pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    ds = NuScenesLSSDataset(split=SPLIT, training=False)
    release_nuscenes()
    loader = DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=False, collate_fn=lss_collate_fn,
                        num_workers=C.NUM_WORKERS)

    res = evaluate(model, loader, CenterPointTargetBuilder(), DetectionLoss().to(device),
                   CenterPointDecoder().to(device), device)

    print(f"Checkpoint : {CHECKPOINT} (epoch {ckpt.get('epoch')}) | split {SPLIT} | {len(ds)} frames")
    print(format_losses("VAL", res["losses"]))
    print(format_report(res["metrics"]))


if __name__ == "__main__":
    main()
