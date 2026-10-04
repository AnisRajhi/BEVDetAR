#!/usr/bin/env python3
"""Test de bout en bout du dataset sur un faux nuScenes réaliste."""
import math

import numpy as np
import pytest
import torch

from lss_det import config as C
from lss_det.data.nuscenes_dataset import NuScenesLSSDataset
from lss_det.models.geometry import Geometry
from tests.fake_nuscenes import FakeNuScenes

ANNS = [
    ("vehicle.car", (12.0, 3.0, 0.9), 0.3, (1.9, 4.5, 1.6), 25),
    ("vehicle.car", (20.0, -4.0, 0.9), 0.0, (1.9, 4.5, 1.6), 0),        # 0 point -> retirée
    ("human.pedestrian.adult", (44.0, 0.0, 0.9), 0.0, (0.7, 0.7, 1.8), 3),  # > 40 m -> retirée
    ("vehicle.car", (44.0, 0.0, 0.9), 1.0, (1.9, 4.5, 1.6), 3),           # car <= 50 m -> gardée
    ("movable_object.barrier", (5.0, 5.0, 0.5), 0.0, (2.0, 0.5, 1.0), 10),  # classe non suivie
]


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    return FakeNuScenes(str(tmp_path_factory.mktemp("nusc")), wall_x=15.0, annotations=ANNS)


def unproject_depth_cells(item, cam_index):
    g = Geometry(C.IMAGE_SIZE, C.DOWNSAMPLE, C.DEPTH_BOUND)
    geom = g(*(item[k][None] for k in ("intrins", "rots", "trans", "post_rots", "post_trans")), item["bda"][None])[0]
    bins = item["depth_bins"][cam_index]
    ii, jj = torch.nonzero(bins >= 0, as_tuple=True)
    return geom[cam_index, bins[ii, jj], ii, jj]          # [P,3] dans le repère BEV (avec bda)


def test_val_sample_shapes_and_gt_filtering(fake):
    ds = NuScenesLSSDataset(split="train", training=False, nusc=fake, version="v1.0-mini")
    item = ds[0]
    assert item["images"].shape == (6, 3, *C.IMAGE_SIZE)
    assert item["depth_bins"].shape == (6, *C.FEATURE_SIZE)
    assert torch.allclose(item["bda"], torch.eye(3))
    # 2 voitures gardées (12 m et 44 m), le reste filtré
    assert item["gt_boxes"].shape[0] == 2
    assert sorted(item["gt_labels"].tolist()) == [0, 0]
    b = item["gt_boxes"][torch.argmin(item["gt_boxes"][:, 0])]
    assert torch.allclose(b[:3], torch.tensor([12.0, 3.0, 0.9]), atol=1e-3)
    assert torch.allclose(b[3:6], torch.tensor([4.5, 1.9, 1.6]), atol=1e-4)  # [l, w, h]
    assert abs(b[6].item() - 0.3) < 1e-4


def test_depth_labels_unproject_onto_the_wall_val(fake):
    ds = NuScenesLSSDataset(split="train", training=False, nusc=fake, version="v1.0-mini")
    item = ds[0]
    pts = unproject_depth_cells(item, C.CAMERAS.index("CAM_FRONT"))
    assert pts.shape[0] > 50
    # Le mur est à x = 15 m : erreur max = demi-bin (0.5 m) le long du rayon
    assert (pts[:, 0] - 15.0).abs().max() < 0.6
    # La caméra arrière ne voit pas le mur
    assert (item["depth_bins"][C.CAMERAS.index("CAM_BACK")] >= 0).sum() == 0


@pytest.mark.parametrize("seed", range(4))
def test_depth_labels_and_gt_follow_augmentations_train(fake, seed):
    torch.manual_seed(seed)
    ds = NuScenesLSSDataset(split="train", training=True, nusc=fake, version="v1.0-mini")
    item = ds[0]
    bda = item["bda"]
    pts = unproject_depth_cells(item, C.CAMERAS.index("CAM_FRONT"))
    assert pts.shape[0] > 30
    pts_ego = pts @ torch.linalg.inv(bda).T
    assert (pts_ego[:, 0] - 15.0).abs().max() < 0.6

    gt_ego = item["gt_boxes"][:, :3] @ torch.linalg.inv(bda).T
    near = gt_ego[torch.argmin(gt_ego[:, 0].abs())]
    assert torch.allclose(near, torch.tensor([12.0, 3.0, 0.9]), atol=1e-3)
