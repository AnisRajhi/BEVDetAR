#!/usr/bin/env python3
"""Partie de trainval : filtrage des fichiers absents, NuScenes partagé puis libéré."""
import copy

import pytest
import torch

from lss_det import config as C
from lss_det.data import nuscenes_dataset as ND
from tests.fake_nuscenes import FakeNuScenes


def add_incomplete_sample(fake):
    """Deuxième frame dont l'image CAM_FRONT n'a pas été téléchargée."""
    s1 = copy.deepcopy(fake.sample[0])
    s1["token"] = "s1"
    s1["anns"] = []
    s1["data"]["CAM_FRONT"] = "sd_missing"
    fake.tables["sample_data"]["sd_missing"] = dict(fake.tables["sample_data"]["sd_CAM_FRONT"])
    fake.paths["sd_missing"] = "/n/existe/pas/CAM_FRONT.jpg"
    fake.sample.append(s1)


def test_frames_with_missing_files_are_skipped(tmp_path, capsys):
    fake = FakeNuScenes(str(tmp_path), scene_name="scene-0061")
    add_incomplete_sample(fake)
    ds = ND.NuScenesLSSDataset(split="train", training=False, nusc=fake, version="v1.0-mini")
    assert len(ds) == 1 and ds.infos[0]["token"] == "s0"
    assert "1/2 frames" in capsys.readouterr().out


def test_filter_can_be_disabled(tmp_path, monkeypatch):
    fake = FakeNuScenes(str(tmp_path), scene_name="scene-0061")
    add_incomplete_sample(fake)
    monkeypatch.setattr(C, "REQUIRE_AVAILABLE_FILES", False)
    ds = ND.NuScenesLSSDataset(split="train", training=False, nusc=fake, version="v1.0-mini")
    assert len(ds) == 2


def test_no_usable_frame_gives_explicit_error(tmp_path):
    fake = FakeNuScenes(str(tmp_path), scene_name="scene-0061")
    fake.paths["sd_CAM_BACK"] = "/n/existe/pas.jpg"
    with pytest.raises(RuntimeError, match="DATAROOT"):
        ND.NuScenesLSSDataset(split="train", training=False, nusc=fake, version="v1.0-mini")


def test_trainval_official_split(tmp_path):
    fake = FakeNuScenes(str(tmp_path), scene_name="scene-0003")     # scène de val officielle
    ds = ND.NuScenesLSSDataset(split="val", training=False, nusc=fake, version="v1.0-trainval")
    assert len(ds) == 1
    with pytest.raises(RuntimeError):
        ND.NuScenesLSSDataset(split="train", training=False, nusc=fake, version="v1.0-trainval")


def test_dataset_works_after_nuscenes_is_gone(tmp_path):
    fake = FakeNuScenes(str(tmp_path), scene_name="scene-0061",
                        annotations=[("vehicle.car", (12.0, 3.0, 0.9), 0.3, (1.9, 4.5, 1.6), 25)])
    ds = ND.NuScenesLSSDataset(split="train", training=False, nusc=fake, version="v1.0-mini")
    assert not any(v is fake for v in vars(ds).values())
    fake.tables.clear()                      # toute consultation de NuScenes échouerait
    item = ds[0]
    assert item["gt_boxes"].shape == (1, 7)
    assert torch.allclose(item["gt_boxes"][0, :3], torch.tensor([12.0, 3.0, 0.9]), atol=1e-3)
    assert (item["depth_bins"] >= 0).sum() > 0


def test_nuscenes_is_loaded_once_then_released(monkeypatch):
    calls = []

    class Dummy:
        def __init__(self, version, dataroot, verbose):
            calls.append((version, dataroot))

    import nuscenes.nuscenes
    monkeypatch.setattr(nuscenes.nuscenes, "NuScenes", Dummy)
    ND.release_nuscenes()
    a = ND.get_nuscenes("v1.0-trainval", "/data/nusc")
    b = ND.get_nuscenes("v1.0-trainval", "/data/nusc/")
    assert a is b and len(calls) == 1
    ND.release_nuscenes()
    assert not ND._NUSC_CACHE
    ND.get_nuscenes("v1.0-trainval", "/data/nusc")
    assert len(calls) == 2
    ND.release_nuscenes()
