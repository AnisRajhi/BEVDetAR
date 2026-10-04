#!/usr/bin/env python3
"""Exécution réelle de audit_dataset et du moteur train/eval sur un faux nuScenes."""
import pytest
import torch
from torch.utils.data import DataLoader

import audit_dataset
from lss_det import config as C
from lss_det.data.collate import lss_collate_fn
from lss_det.data.nuscenes_dataset import NuScenesLSSDataset
from lss_det.decoding.decode_boxes import CenterPointDecoder
from lss_det.engine import evaluate, make_scheduler, train_one_epoch
from lss_det.losses.detection_loss import DetectionLoss
from lss_det.models.lss_detector import LSSDetector
from lss_det.targets.centerpoint_targets import CenterPointTargetBuilder
from tests.fake_nuscenes import FakeNuScenes

ANNS = [
    ("vehicle.car", (12.0, 3.0, 0.9), 0.3, (1.9, 4.5, 1.6), 25),     # apprenable
    ("vehicle.car", (20.0, -4.0, 0.9), 0.0, (1.9, 4.5, 1.6), 0),     # 0 point
    ("human.pedestrian.adult", (44.0, 0.0, 0.9), 0.0, (0.7, 0.7, 1.8), 3),  # > 40 m
    ("vehicle.car", (48.0, 1.0, 0.9), 0.0, (1.9, 4.5, 1.6), 3),      # v1 : profondeur > 45
    ("vehicle.car", (49.0, 30.0, 0.9), 0.0, (1.9, 4.5, 1.6), 3),     # 57 m : hors portée
]


@pytest.fixture(scope="module")
def fakes(tmp_path_factory):
    return {
        "train": FakeNuScenes(str(tmp_path_factory.mktemp("tr")), scene_name="scene-0061", annotations=ANNS),
        "val": FakeNuScenes(str(tmp_path_factory.mktemp("va")), scene_name="scene-0103", annotations=ANNS),
    }


def test_audit_counts(fakes):
    res = audit_dataset.main(fakes)
    old, new = res["train"]["old"], res["train"]["new"]
    assert old["car"]["targets"] == 4
    assert old["car"]["empty"] == 1
    assert old["car"]["beyond_eval_range"] == 1
    assert old["car"]["visible_but_depth_out"] >= 1
    assert old["car"]["learnable"] == 1
    assert new["car"]["targets"] == 2 and new["car"]["reachable"] == 2
    assert old["pedestrian"]["beyond_eval_range"] == 1 and new["pedestrian"]["targets"] == 0


def test_engine_train_and_eval_one_step(fakes):
    torch.manual_seed(0)
    tr = NuScenesLSSDataset(split="train", training=True, nusc=fakes["train"])
    va = NuScenesLSSDataset(split="val", training=False, nusc=fakes["val"])
    tl = DataLoader(tr, batch_size=1, collate_fn=lss_collate_fn)
    vl = DataLoader(va, batch_size=1, collate_fn=lss_collate_fn)
    model = LSSDetector(camera_pretrained=False)
    opt = torch.optim.AdamW(model.parameter_groups(2e-4, 0.1, 1e-2), lr=2e-4)
    sched = make_scheduler(opt, total_steps=10, warmup_steps=2)
    stats = train_one_epoch(model, tl, CenterPointTargetBuilder(), DetectionLoss(), opt, sched, "cpu", accum_steps=8)
    assert torch.isfinite(torch.tensor(stats["loss"])) and stats["loss_depth"] > 0
    assert 0 <= stats["depth_acc1"] <= 1
    res = evaluate(model, vl, CenterPointTargetBuilder(), DetectionLoss(), CenterPointDecoder(), "cpu")
    assert res["metrics"]["num_samples"] == 1
    assert res["metrics"]["per_class"]["car"]["num_gt"] == 2
