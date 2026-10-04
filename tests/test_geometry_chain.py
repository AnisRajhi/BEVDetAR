#!/usr/bin/env python3
"""
Tests de la chaîne géométrique complète. Si l'un d'eux casse, le
réseau apprend sur des données incohérentes : à lancer après toute
modification de transforms / geometry / depth_targets / bda.
"""
import math

import numpy as np
import pytest
import torch
from PIL import Image

from lss_det import config as C
from lss_det.data.bev_augmentation import apply_bda_to_boxes, build_bda_matrix
from lss_det.data.depth_targets import build_depth_target
from lss_det.data.transforms import LSSImageAugmentation
from lss_det.models.geometry import Geometry

OPTICAL_TO_EGO = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])
K = np.array([[1266.0, 0, 816.0], [0, 1266.0, 491.0], [0, 0, 1.0]])


def test_post_transform_matches_pil():
    """post_rot/post_trans doit suivre exactement le déplacement réel des pixels."""
    aug = LSSImageAugmentation(C.IMAGE_SIZE, training=True, aug=C.IMG_AUG)
    rng = np.random.default_rng(1)
    W, H = 1600, 900
    yy, xx = np.mgrid[0:H, 0:W]
    checked = 0
    for _ in range(30):
        u0, v0 = rng.uniform(300, 1300), rng.uniform(450, 850)
        blob = np.exp(-((xx - u0) ** 2 + (yy - v0) ** 2) / (2 * 12.0 ** 2))
        img = Image.fromarray((np.stack([blob] * 3, -1) * 255).astype(np.uint8))
        p = aug.sample_params(W, H, rng)
        R, t = aug.compute_post_transform((W, H), p)
        out = np.asarray(aug.apply_to_image(img, p)).astype(float)[..., 0]
        pred = R[:2, :2].numpy() @ np.array([u0, v0]) + t[:2].numpy()
        if not (20 < pred[0] < C.IMAGE_SIZE[1] - 20 and 20 < pred[1] < C.IMAGE_SIZE[0] - 20):
            continue
        w = out ** 2
        ys, xs = np.mgrid[0:out.shape[0], 0:out.shape[1]]
        c = np.array([(w * xs).sum() / w.sum(), (w * ys).sum() / w.sum()])
        assert np.linalg.norm(c - pred) < 0.1
        checked += 1
    assert checked >= 5


def test_frustum_is_cell_and_bin_centered():
    g = Geometry(C.IMAGE_SIZE, C.DOWNSAMPLE, C.DEPTH_BOUND)
    f = g.frustum
    s = C.DOWNSAMPLE
    assert torch.allclose(f[0, 0, 0], torch.tensor([(s - 1) / 2, (s - 1) / 2, C.DEPTH_BOUND[0] + C.DEPTH_BOUND[2] / 2]))
    assert torch.allclose(f[0, 0, 1, 0] - f[0, 0, 0, 0], torch.tensor(float(s)))
    assert f.shape == (C.NUM_DEPTH_BINS, *C.FEATURE_SIZE, 3)


@pytest.mark.parametrize("seed", range(5))
def test_projection_unprojection_roundtrip_with_all_augmentations(seed):
    """
    ego -> caméra -> pixel original -> pixel augmenté -> Geometry.image_to_ego -> BEV
    doit redonner bda @ P exactement.
    """
    rng = np.random.default_rng(seed)
    R = (np.array([[math.cos(0.7), -math.sin(0.7), 0], [math.sin(0.7), math.cos(0.7), 0], [0, 0, 1]]) @ OPTICAL_TO_EGO)
    t = np.array([1.2, 0.4, 1.6])
    aug = LSSImageAugmentation(C.IMAGE_SIZE, training=True, aug=C.IMG_AUG)
    p = aug.sample_params(1600, 900, rng)
    post_rot, post_trans = aug.compute_post_transform((1600, 900), p)
    bda = build_bda_matrix(rng.uniform(-22, 22), rng.uniform(0.95, 1.05), bool(rng.random() < .5), bool(rng.random() < .5))

    pts_cam = np.stack([rng.uniform(-5, 5, 50), rng.uniform(-1, 2, 50), rng.uniform(3, 50, 50)], 1)
    pts_ego = pts_cam @ R.T + t
    uv = (pts_cam @ K.T)[:, :2] / pts_cam[:, 2:3]
    uv = uv @ post_rot[:2, :2].numpy().T + post_trans[:2].numpy()
    uvd = torch.from_numpy(np.concatenate([uv, pts_cam[:, 2:3]], 1)).float()[None, None]

    f = lambda a: torch.from_numpy(np.asarray(a)).float()[None, None]
    out = Geometry.image_to_ego(uvd, f(K), f(R), f(t), post_rot[None, None], post_trans[None, None], bda[None])
    expected = torch.from_numpy(pts_ego).float() @ bda.T
    assert torch.allclose(out[0, 0], expected, atol=2e-3)


def test_depth_target_cell_and_bin_are_consistent_with_frustum():
    g = Geometry(C.IMAGE_SIZE, C.DOWNSAMPLE, C.DEPTH_BOUND)
    aug = LSSImageAugmentation(C.IMAGE_SIZE, training=False)
    post_rot, post_trans = aug.compute_post_transform((1600, 900), aug.sample_params(1600, 900, None))
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = np.array([[rng.uniform(-8, 8), rng.uniform(-1, 1.5), rng.uniform(2, 55)]])
        bins, dmin = build_depth_target(p, K, post_rot.numpy(), post_trans.numpy(), C.IMAGE_SIZE, C.DOWNSAMPLE, C.DEPTH_BOUND)
        idx = np.argwhere(bins >= 0)
        if idx.size == 0:
            continue
        i, j = idx[0]
        k = bins[i, j]
        uv = (K @ p[0])[:2] / p[0, 2]
        uv = post_rot[:2, :2].numpy() @ uv + post_trans[:2].numpy()
        center = g.frustum[k, i, j].numpy()
        assert abs(center[0] - uv[0]) <= C.DOWNSAMPLE / 2 + 1e-4
        assert abs(center[1] - uv[1]) <= C.DOWNSAMPLE / 2 + 1e-4
        assert abs(center[2] - p[0, 2]) <= C.DEPTH_BOUND[2] / 2 + 1e-4


def test_depth_target_keeps_nearest_point_per_cell():
    aug = LSSImageAugmentation(C.IMAGE_SIZE, training=False)
    pr, pt = aug.compute_post_transform((1600, 900), aug.sample_params(1600, 900, None))
    ray = np.array([0.0, 0.02, 1.0])          # 3 points sur le même rayon
    p = np.stack([ray * 30.0, ray * 12.0, ray * 20.0])
    bins, dmin = build_depth_target(p, K, pr.numpy(), pt.numpy(), C.IMAGE_SIZE, C.DOWNSAMPLE, C.DEPTH_BOUND)
    assert (bins >= 0).sum() == 1
    assert np.isclose(dmin[bins >= 0][0], 12.0)


@pytest.mark.parametrize("flip_x,flip_y", [(False, False), (True, False), (False, True), (True, True)])
def test_bda_on_boxes_matches_bda_on_corners(flip_x, flip_y):
    def corners(b):
        x, y, z, l, w, h, yaw = b.tolist()
        c, s = math.cos(yaw), math.sin(yaw)
        out = []
        for dx in (-l / 2, l / 2):
            for dy in (-w / 2, w / 2):
                for dz in (-h / 2, h / 2):
                    out.append([x + c * dx - s * dy, y + s * dx + c * dy, z + dz])
        return torch.tensor(out)

    bda = build_bda_matrix(17.0, 1.03, flip_x, flip_y)
    boxes = torch.tensor([[10.0, -3.0, 0.8, 4.5, 1.9, 1.6, 0.4], [-20.0, 7.0, 1.0, 11.0, 2.9, 3.5, -2.0]])
    new = apply_bda_to_boxes(boxes, bda)
    for b, nb in zip(boxes, new):
        a = corners(b) @ bda.T
        e = corners(nb)
        # même ensemble de coins (ordre éventuellement permuté par le miroir)
        d = torch.cdist(a, e).min(dim=1).values
        assert d.max() < 1e-4
