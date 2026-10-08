#!/usr/bin/env python3
"""analyze_errors.py : catégories d'erreurs, décomposition radiale, limite structurelle, bout en bout."""
import numpy as np
import torch

import analyze_errors as AE
from lss_det import config as C
from lss_det.decoding.decode_boxes import CenterPointDecoder
from lss_det.metrics.detection_metrics import DetectionEvaluator
from lss_det.targets.centerpoint_targets import CenterPointTargetBuilder

CAR, TRUCK, BUS, PED = (C.CLASSES.index(c) for c in ("car", "truck", "bus", "pedestrian"))


def test_categorize_frame_four_fates():
    gt = np.array([[10.0, 0, 0, 4.5, 1.9, 1.6, 0], [20.0, 5, 0, 8, 2.5, 3, 0],
                   [5.0, -3, 0, 0.7, 0.7, 1.8, 0], [30.0, 10, 0, 11, 2.9, 3.4, 0]])
    gl = np.array([CAR, TRUCK, PED, BUS])
    pred = np.array([[10.5, 0, 0, 4.5, 1.9, 1.6, 0],     # voiture juste
                     [20.3, 5, 0, 4.5, 1.9, 1.6, 0],     # camion vu comme voiture
                     [8.0, -3, 0, 0.7, 0.7, 1.8, 0]])    # piéton placé à 3 m
    fate = AE.categorize_frame(pred, np.array([0.9, 0.8, 0.7]), np.array([CAR, CAR, PED]), gt, gl)
    assert [f[0] for f in fate] == ["bonne classe <2 m", "autre classe <2 m", "détecté 2-4 m", "non vu"]
    assert fate[1][1] == CAR and fate[3][1] == -1


def test_radial_lateral():
    r, l = AE.radial_lateral(np.array([11.0, 0.5]), np.array([10.0, 0.0]))
    assert np.isclose(r, 1.0) and np.isclose(l, 0.5)


def test_oracle_same_cell_is_lost_neighbour_is_kept():
    b = CenterPointTargetBuilder()
    gt = torch.tensor([[10.1, 2.1, 0.9, 0.7, 0.7, 1.8, 0.0],    # piéton A, case (76, 66)
                       [10.3, 2.3, 0.9, 0.7, 0.7, 1.8, 0.0],    # piéton B, même case que A
                       [12.1, 5.0, 0.9, 0.7, 0.7, 1.8, 0.0],    # piéton C
                       [12.9, 5.0, 0.9, 0.7, 0.7, 1.8, 0.0]])   # piéton D, case voisine de C
    gl = torch.tensor([PED] * 4)
    same, nb = AE.crowding(gt.numpy(), gl.numpy(), b)
    assert same.tolist() == [True, True, False, False] and nb.tolist() == [False, False, True, True]
    det = CenterPointDecoder()(AE.perfect_predictions(b([gt], [gl])))[0]
    ev = DetectionEvaluator()
    ev.add("s", det["boxes"], det["scores"], det["labels"], gt, gl)
    assert abs(ev.compute()["per_class"]["pedestrian"]["max_recall"] - 0.75) < 1e-6   # 1 sur 4 perdu


def test_analyze_end_to_end(tmp_path, monkeypatch):
    from lss_det.data.nuscenes_dataset import NuScenesLSSDataset as RealDS
    from lss_det.engine import config_snapshot
    from lss_det.models.lss_detector import LSSDetector
    from tests.fake_nuscenes import FakeNuScenes
    fake = FakeNuScenes(str(tmp_path / "n"), "scene-0103",
                        annotations=[("vehicle.car", (12.0, 3.0, 0.9), 0.3, (1.9, 4.5, 1.6), 25),
                                     ("human.pedestrian.adult", (7.0, 1.5, 0.9), 0.0, (0.7, 0.7, 1.8), 10)])
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": LSSDetector(camera_pretrained=False).state_dict(),
                "config": config_snapshot(), "epoch": 0}, ckpt)
    monkeypatch.setattr(AE, "NuScenesLSSDataset",
                        lambda split, training: RealDS(split=split, training=training, nusc=fake, version="v1.0-mini"))
    monkeypatch.setattr(AE, "CHECKPOINT", str(ckpt))
    monkeypatch.setattr(AE, "REPORT_PATH", str(tmp_path / "report.txt"))
    monkeypatch.setattr(AE, "DEVICE", "cpu")
    monkeypatch.setattr(C, "NUM_WORKERS", 0)
    AE.main()
    text = (tmp_path / "report.txt").read_text()
    for section in ("1. LIMITE STRUCTURELLE", "a. Devenir", "b. Confusion", "c. Part d'objets NON VUS",
                    "d. Erreurs de cap", "e. Erreur de centre"):
        assert section in text
    assert "car" in text and "pedestrian" in text
