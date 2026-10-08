#!/usr/bin/env python3
"""
v2.3 : alignement spatial des ré-agrandissements et décodage JPEG réduit.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from lss_det import config as C
from lss_det.data.transforms import LSSImageAugmentation
from lss_det.models.bev_backbone import BEVBackbone, upsample_aligned


def down(x):
    """Convolution centrée de stride 2 (comme le stem 7x7/p3 et les BasicBlocks 3x3/p1)."""
    k = torch.tensor([[[[0.0, 0.0, 0.0], [0.25, 0.5, 0.25], [0.0, 0.0, 0.0]]]])
    return F.conv2d(x, k, stride=2, padding=1)


def centroid_x(y):
    y = y.clamp(min=0)[0, 0].sum(0)
    return float((y * torch.arange(y.numel())).sum() / y.sum())


def errors(upsample):
    fine, ctx = [], []
    for j in range(8, 120, 8):
        x = torch.zeros(1, 1, 8, 128)
        x[..., j] = 1.0
        x64 = down(x)
        fine.append(centroid_x(upsample(x64, (8, 128))) - j)
        x16 = down(down(x64))
        ctx.append(centroid_x(upsample(upsample(x16, (4, 64)), (8, 128))) - j)
    return np.array(fine), np.array(ctx)


def test_aligned_upsampling_has_no_position_dependent_shift():
    fine, ctx = errors(upsample_aligned)
    assert np.abs(fine).max() < 0.05
    assert np.abs(ctx).max() < 0.05


def test_align_corners_true_shift_depended_on_position():
    """Documente le défaut corrigé : le décalage variait avec la position."""
    ac = lambda x, size: F.interpolate(x, size=size, mode="bilinear", align_corners=True)
    fine, ctx = errors(ac)
    assert np.ptp(fine) > 0.5 and np.ptp(ctx) > 3.0


def test_upsample_aligned_shapes_and_bev_backbone():
    x = torch.randn(2, 5, 16, 16)
    assert upsample_aligned(x, (64, 64)).shape == (2, 5, 64, 64)
    with pytest.raises(ValueError):
        upsample_aligned(x, (60, 64))
    out = BEVBackbone(64, 128)(torch.randn(1, 64, C.NX, C.NY))
    assert out.shape == (1, 128, C.NX, C.NY)


@pytest.mark.parametrize("seed", range(6))
def test_reduced_jpeg_decoding_keeps_exact_geometry(tmp_path, seed):
    rng = np.random.default_rng(seed)
    W, H = 1600, 900
    u0, v0 = rng.uniform(300, 1300), rng.uniform(450, 850)
    yy, xx = np.mgrid[0:H, 0:W]
    blob = np.exp(-((xx - u0) ** 2 + (yy - v0) ** 2) / (2 * 14.0 ** 2))
    path = tmp_path / "blob.jpg"
    Image.fromarray((np.stack([blob] * 3, -1) * 255).astype(np.uint8)).save(path, quality=95)

    aug = LSSImageAugmentation(C.IMAGE_SIZE, training=True, aug=dict(C.IMG_AUG, rand_flip=True))
    params = aug.sample_params(W, H, rng)
    post_rot, post_trans = aug.compute_post_transform((W, H), params)

    img = Image.open(path)
    out = np.asarray(aug.geometric(img, params)).astype(float)[..., 0]
    assert img.size[0] < W, "le décodeur JPEG aurait dû réduire l'image"

    pred = post_rot[:2, :2].numpy() @ np.array([u0, v0]) + post_trans[:2].numpy()
    if not (25 < pred[0] < C.IMAGE_SIZE[1] - 25 and 25 < pred[1] < C.IMAGE_SIZE[0] - 25):
        pytest.skip("tache hors de l'image recadrée pour ce tirage")
    w = np.clip(out - 10, 0, None) ** 2
    ys, xs = np.mgrid[0:out.shape[0], 0:out.shape[1]]
    c = np.array([(w * xs).sum() / w.sum(), (w * ys).sum() / w.sum()])
    assert np.linalg.norm(c - pred) < 0.2
