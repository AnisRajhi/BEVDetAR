#!/usr/bin/env python3
"""
Entraînement LSS v2 sur nuScenes.

    python3 train.py

Tous les paramètres sont dans lss_det/config.py.

Sélection du meilleur checkpoint sur le mAP de validation, plus sur la
loss de validation : la loss focale de validation peut remonter (le
modèle devient plus confiant, donc chaque erreur coûte plus cher) alors
que la détection continue de s'améliorer. Seule une métrique de
détection dit si le modèle A détecte mieux que le modèle B.

Sanity check : OVERFIT_NUM_SAMPLES = 10 dans config.py -> entraînement
sans augmentation sur 10 frames, évaluées sur ces mêmes 10 frames
(batch 1, sans accumulation, OVERFIT_EPOCHS epochs). mAP élevé et
dAcc proche de 1 attendus ; sinon, la chaîne est cassée.
"""

import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from lss_det import config as C
from lss_det.data.collate import lss_collate_fn
from lss_det.data.nuscenes_dataset import NuScenesLSSDataset, release_nuscenes
from lss_det.decoding.decode_boxes import CenterPointDecoder
from lss_det.engine import (config_snapshot, evaluate, format_losses, make_scheduler,
                            seed_everything, train_one_epoch)
from lss_det.losses.detection_loss import DetectionLoss
from lss_det.metrics.detection_metrics import format_report
from lss_det.models.lss_detector import LSSDetector
from lss_det.targets.centerpoint_targets import CenterPointTargetBuilder


def main():
    seed_everything(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(C.CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LSS DETECTOR v2 -", C.VERSION)
    print("=" * 70)
    print("Device :", device, f"({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    print("Config :", C.summary())

    train_ds = NuScenesLSSDataset(split="train", training=True)
    overfit = C.OVERFIT_NUM_SAMPLES > 0
    if overfit:
        train_ds.disable_augmentation = True
        idx = list(range(min(C.OVERFIT_NUM_SAMPLES, len(train_ds))))
        train_set, val_set = Subset(train_ds, idx), Subset(train_ds, idx)
        print(f"MODE OVERFIT : {len(idx)} frames, sans augmentation")
    else:
        train_set = train_ds
        val_set = NuScenesLSSDataset(split="val", training=False)

    # Les datasets ont extrait leurs infos : on libère NuScenes AVANT de
    # créer les workers du DataLoader (sinon chaque worker en hérite).
    release_nuscenes()

    epochs = C.OVERFIT_EPOCHS if overfit else C.EPOCHS
    accum = 1 if overfit else C.ACCUM_STEPS
    warmup = C.OVERFIT_WARMUP_STEPS if overfit else C.WARMUP_STEPS
    eval_every = C.OVERFIT_EVAL_EVERY if overfit else C.EVAL_EVERY

    print(f"Train : {len(train_set)} | Val : {len(val_set)}")

    loader_kw = dict(collate_fn=lss_collate_fn, num_workers=C.NUM_WORKERS,
                     pin_memory=device.type == "cuda", persistent_workers=C.NUM_WORKERS > 0)
    train_loader = DataLoader(train_set, batch_size=C.BATCH_SIZE, shuffle=True, drop_last=False, **loader_kw)
    val_loader = DataLoader(val_set, batch_size=C.BATCH_SIZE, shuffle=False, drop_last=False, **loader_kw)

    model = LSSDetector().to(device)
    criterion = DetectionLoss().to(device)
    target_builder = CenterPointTargetBuilder()
    decoder = CenterPointDecoder().to(device)

    groups = model.parameter_groups(C.LEARNING_RATE, C.BACKBONE_LR_MULT, C.WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(groups, lr=C.LEARNING_RATE)
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    scheduler = make_scheduler(optimizer, total_steps=steps_per_epoch * epochs, warmup_steps=warmup)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Paramètres : {n_params:.2f} M | " + " | ".join(
        f"{g['name']}: {sum(p.numel() for p in g['params']) / 1e6:.2f} M @ lr {g['lr']:.1e}" for g in groups))
    print(f"Batch effectif : {C.BATCH_SIZE * accum} | pas d'optimiseur / epoch : {steps_per_epoch} "
          f"| total : {steps_per_epoch * epochs} | warmup : {warmup}")
    print()

    best_map = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, target_builder, criterion, optimizer, scheduler, device,
                                      accum_steps=accum)
        print(f"Epoch {epoch:03d}/{epochs}  ({time.time() - t0:.0f} s, lr={train_stats['lr']:.2e})")
        print(format_losses("TRAIN", train_stats))

        record = {"epoch": epoch, "train": train_stats}
        if epoch % eval_every == 0 or epoch == epochs:
            val = evaluate(model, val_loader, target_builder, criterion, decoder, device)
            print(format_losses("VAL", val["losses"]))
            print("  " + format_report(val["metrics"]).replace("\n", "\n  "))
            record["val_losses"] = val["losses"]
            record["val_mAP"] = val["metrics"]["mAP"]

            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val": {"losses": val["losses"], "mAP": val["metrics"]["mAP"],
                        "ATE": val["metrics"]["ATE"], "ASE": val["metrics"]["ASE"], "AOE": val["metrics"]["AOE"]},
                "config": config_snapshot(),
            }
            torch.save(state, ckpt_dir / "last.pt")          # avec optimiseur : reprise possible
            if val["metrics"]["mAP"] > best_map:
                best_map = val["metrics"]["mAP"]
                light = {k: v for k, v in state.items() if k != "optimizer_state_dict"}
                torch.save(light, ckpt_dir / "best_map.pt")  # poids seuls (~3x plus léger)
                print(f"  -> nouveau meilleur mAP : {best_map:.4f}")

        history.append(record)
        torch.save(history, ckpt_dir / "history.pt")
        print()

    print("=" * 70)
    print(f"Meilleur mAP val : {best_map:.4f}  ->  {ckpt_dir / 'best_map.pt'}")


if __name__ == "__main__":
    main()
