#!/usr/bin/env python3
"""
Miroir cohérent : l'apparence des objets doit rester compatible avec leur
orientation cible après augmentation.

Invariant testé : pour chaque caméra, l'application linéaire
    (u'·d, v'·d, d) image augmentée  ->  point BEV augmenté
    A = bda · R · K⁻¹ · post_rot⁻¹
doit avoir un déterminant POSITIF, comme une caméra non augmentée. C'est la
condition pour qu'une caméra virtuelle « normale » (sans miroir) produise
exactement ces images d'un monde éventuellement miroité. Avec des miroirs
tirés indépendamment, l'invariant est violé dans ~50 % des cas : la cible
d'orientation contredit alors l'apparence et la tête rot n'apprend rien.
"""
import numpy as np
import pytest
import torch

from lss_det import config as C
from lss_det.data.bev_augmentation import sample_bda
from lss_det.data.nuscenes_dataset import NuScenesLSSDataset
from lss_det.data.transforms import LSSImageAugmentation
from tests.fake_nuscenes import FakeNuScenes

LEGACY_AUG = dict(C.IMG_AUG, rand_flip=True)
LEGACY_BDA = dict(C.BDA, flip_x_prob=0.5, flip_y_prob=0.5)


def handedness(post_rot, bda):
    return float(np.linalg.det(post_rot[:2, :2].numpy()) * np.linalg.det(bda.numpy()))


def test_coherent_mirror_keeps_handedness():
    aug = LSSImageAugmentation(C.IMAGE_SIZE, training=True, aug=LEGACY_AUG)
    seen_mirror = set()
    for seed in range(300):
        rng = np.random.default_rng(seed)
        mirror = bool(rng.random() < 0.5)
        p = aug.sample_params(1600, 900, rng, force_flip=mirror)
        post_rot, _ = aug.compute_post_transform((1600, 900), p)
        bda, bp = sample_bda(rng, LEGACY_BDA, training=True, mirror=mirror)
        assert p["flip"] == mirror
        assert handedness(post_rot, bda) > 0
        # miroir gauche <-> droite uniquement : l'avant de l'ego reste vers +x
        assert bp["flip_x"] is False and bp["flip_y"] == mirror
        assert float((bda @ torch.tensor([1.0, 0.0, 0.0]))[0]) > 0
        seen_mirror.add(mirror)
    assert seen_mirror == {True, False}


def test_independent_flips_break_handedness_half_the_time():
    """Documente le défaut de la v2/v2.1 (miroirs indépendants)."""
    aug = LSSImageAugmentation(C.IMAGE_SIZE, training=True, aug=LEGACY_AUG)
    bad = 0
    for seed in range(400):
        rng = np.random.default_rng(seed)
        p = aug.sample_params(1600, 900, rng)
        post_rot, _ = aug.compute_post_transform((1600, 900), p)
        bda, _ = sample_bda(rng, LEGACY_BDA, training=True)
        bad += handedness(post_rot, bda) < 0
    assert 0.35 < bad / 400 < 0.65


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    return FakeNuScenes(str(tmp_path_factory.mktemp("nusc")), scene_name="scene-0061",
                        annotations=[("vehicle.car", (12.0, 3.0, 0.9), 0.3, (1.9, 4.5, 1.6), 25)])


def test_dataset_all_cameras_coherent(fake, monkeypatch):
    monkeypatch.setattr(C, "MIRROR_PROB", 0.5)
    monkeypatch.setattr(C, "IMG_AUG", dict(C.IMG_AUG, rand_flip=False))    # doit être ignoré
    monkeypatch.setattr(C, "BDA", dict(C.BDA, flip_x_prob=0.0, flip_y_prob=0.0))
    ds = NuScenesLSSDataset(split="train", training=True, nusc=fake, version="v1.0-mini")
    mirrored = set()
    for seed in range(16):
        torch.manual_seed(seed)
        item = ds[0]
        bda = item["bda"]
        flips = set()
        for k in range(len(C.CAMERAS)):
            A = bda @ item["rots"][k] @ torch.linalg.inv(item["intrins"][k]) @ torch.linalg.inv(item["post_rots"][k])
            assert torch.linalg.det(A) > 0, f"caméra {C.CAMERAS[k]} : apparence et cible incohérentes"
            flips.add(bool(torch.linalg.det(item["post_rots"][k][:2, :2]) < 0))
        assert len(flips) == 1                        # les 6 images partagent le même miroir
        assert float((bda @ torch.tensor([1.0, 0.0, 0.0]))[0]) > 0   # avant de l'ego vers +x
        mirrored |= flips
        # la boîte suit toujours la BDA (centre + cap)
        gt_ego = item["gt_boxes"][0, :3] @ torch.linalg.inv(bda).T
        assert torch.allclose(gt_ego, torch.tensor([12.0, 3.0, 0.9]), atol=1e-3)
    assert mirrored == {True, False}


def test_mirror_disabled_and_validation(fake, monkeypatch):
    monkeypatch.setattr(C, "MIRROR_PROB", 0.0)
    ds = NuScenesLSSDataset(split="train", training=True, nusc=fake, version="v1.0-mini")
    for seed in range(4):
        torch.manual_seed(seed)
        item = ds[0]
        assert torch.linalg.det(item["bda"]) > 0
        assert all(torch.linalg.det(r[:2, :2]) > 0 for r in item["post_rots"])
    val = NuScenesLSSDataset(split="train", training=False, nusc=fake, version="v1.0-mini")
    assert torch.allclose(val[0]["bda"], torch.eye(3))
