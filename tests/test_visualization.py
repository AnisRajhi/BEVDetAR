#!/usr/bin/env python3
"""Outils de visualisation : géométrie des boîtes, projection, appariement, rendu de bout en bout."""
import math
import os

import numpy as np
import torch

from lss_det import config as C
from lss_det.visualization.draw import box_corners, match_frame, project_to_image


def test_box_corners_geometry():
    c = box_corners([10.0, 2.0, 1.0, 4.0, 2.0, 1.5, 0.0])
    assert np.allclose(c[:, 0].min(), 8.0) and np.allclose(c[:, 0].max(), 12.0)
    assert np.allclose(c[:, 1].min(), 1.0) and np.allclose(c[:, 1].max(), 3.0)
    assert np.allclose(c[:4, 2], 0.25) and np.allclose(c[4:, 2], 1.75)
    assert np.allclose(c[:2, 0], 12.0)                           # coins 0-1 = face avant
    r = box_corners([0.0, 0.0, 0.0, 4.0, 2.0, 1.0, math.pi / 2])
    assert np.allclose(r[:2, 1], 2.0)                             # cap 90° : l'avant pointe vers +y


def test_projection_point_ahead_lands_on_principal_point():
    R = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])     # caméra avant (optique -> ego)
    t = np.array([1.5, 0.0, 1.5])
    K = np.array([[1266.0, 0, 816.0], [0, 1266.0, 491.0], [0, 0, 1]])
    uv, z = project_to_image(np.array([[11.5, 0.0, 1.5], [11.5, 1.0, 1.5]]), R, t, K)
    assert np.allclose(uv[0], [816.0, 491.0]) and np.isclose(z[0], 10.0)
    assert uv[1, 0] < 816.0                                       # un point à gauche apparaît à gauche


def test_match_frame():
    gt = np.array([[10.0, 0, 0, 4, 2, 1.5, 0], [20.0, 5, 0, 4, 2, 1.5, 0], [5.0, -3, 0, 0.7, 0.7, 1.8, 0]])
    gl = np.array([0, 0, 3])
    pred = np.array([[10.5, 0, 0, 4, 2, 1.5, 0],      # juste
                     [20.0, 8, 0, 4, 2, 1.5, 0],      # 3 m trop loin : fausse
                     [5.2, -3, 0, 0.7, 0.7, 1.8, 0]])  # bonne position mais mauvaise classe : fausse
    tp, found = match_frame(pred, [0.9, 0.8, 0.7], np.array([0, 0, 0]), gt, gl)
    assert tp.tolist() == [True, False, False] and found.tolist() == [True, False, False]


def test_visualize_end_to_end(tmp_path):
    from lss_det.data.nuscenes_dataset import NuScenesLSSDataset as RealDS
    from lss_det.engine import config_snapshot
    from lss_det.models.lss_detector import LSSDetector
    from tests.fake_nuscenes import FakeNuScenes
    import visualize as V
    fake = FakeNuScenes(str(tmp_path / "n"), "scene-0103",
                        annotations=[("vehicle.car", (12.0, 3.0, 0.9), 0.3, (1.9, 4.5, 1.6), 25)])
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": LSSDetector(camera_pretrained=False).state_dict(),
                "config": config_snapshot(), "epoch": 0}, ckpt)
    old = (V.NuScenesLSSDataset, V.CHECKPOINT, V.OUTPUT_DIR, V.MODE, V.DEVICE)
    try:
        V.NuScenesLSSDataset = lambda split, training: RealDS(split=split, training=training, nusc=fake,
                                                              version="v1.0-mini")
        V.CHECKPOINT, V.OUTPUT_DIR, V.MODE, V.DEVICE = str(ckpt), str(tmp_path / "out"), "scene", "cpu"
        V.main()
    finally:
        V.NuScenesLSSDataset, V.CHECKPOINT, V.OUTPUT_DIR, V.MODE, V.DEVICE = old
    files = sorted(os.listdir(tmp_path / "out"))
    assert files == ["depth_00000.png", "frame_00000.png", "scene_000.gif"]
    assert all(os.path.getsize(tmp_path / "out" / f) > 10_000 for f in files)
