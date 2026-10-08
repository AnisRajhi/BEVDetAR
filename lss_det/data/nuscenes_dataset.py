#!/usr/bin/env python3
"""
nuScenes -> LSSDetector.

Changements majeurs par rapport à la v1
---------------------------------------
1. Filtrage des GT :
     - boîtes sans aucun point lidar ni radar (objets totalement occultés)
     - boîtes au-delà de la portée d'évaluation de leur classe
   -> on ne demande plus l'impossible au détecteur.

2. Augmentation image géométrique (resize / crop / flip / rotation),
   propagée exactement dans post_rot / post_trans.

3. BEV Data Augmentation : matrice `bda` [3,3] renvoyée au modèle et
   appliquée aux GT.

4. Labels de profondeur lidar par caméra, à la résolution des features :
   `depth_bins` [N, Hf, Wf] (-1 = pas de label).

5. Infos compactes par frame (v2.1) : tout ce dont __getitem__ a besoin
   (chemins, calibrations, poses, annotations) est extrait UNE fois à la
   construction. L'objet NuScenes n'est pas conservé : en trainval, il
   décrit 850 scènes même si une seule archive est téléchargée, et les
   workers du DataLoader (fork) finiraient par en dupliquer la mémoire.

6. Seules les frames dont les 6 images (et le lidar) existent sur le
   disque sont gardées : on peut entraîner sur une partie de trainval.

Sortie d'un sample
------------------
images      [N,3,H,W]
intrins     [N,3,3]
rots        [N,3,3]   caméra -> ego de référence (SANS bda)
trans       [N,3]
post_rots   [N,3,3]
post_trans  [N,3]
bda         [3,3]
depth_bins  [N,Hf,Wf]
gt_boxes    [M,7]     [x,y,z,l,w,h,yaw] dans le repère BEV (AVEC bda)
gt_labels   [M]
sample_token
"""

import gc
import math
import os
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset

from lss_det import config as C
from lss_det.data.bev_augmentation import apply_bda_to_boxes, sample_bda
from lss_det.data.depth_targets import build_depth_target, transform_points
from lss_det.data.transforms import LSSImageAugmentation


def map_category(category_name: str) -> Optional[str]:
    if category_name == "vehicle.car":
        return "car"
    if category_name == "vehicle.truck":
        return "truck"
    if category_name.startswith("vehicle.bus."):
        return "bus"
    if category_name.startswith("human.pedestrian."):
        return "pedestrian"
    if category_name == "vehicle.bicycle":
        return "bicycle"
    if category_name == "vehicle.motorcycle":
        return "motorcycle"
    return None


# ==================================================================
# Chargement partagé de NuScenes
# ==================================================================

_NUSC_CACHE = {}


def get_nuscenes(version: str = C.VERSION, dataroot: str = C.DATAROOT, verbose: bool = False):
    """Un seul chargement par (version, dataroot), partagé par train/val/audit."""
    key = (version, os.path.abspath(os.path.expanduser(dataroot)))
    if key not in _NUSC_CACHE:
        from nuscenes.nuscenes import NuScenes
        _NUSC_CACHE[key] = NuScenes(version=version, dataroot=key[1], verbose=verbose)
    return _NUSC_CACHE[key]


def release_nuscenes():
    """
    Libère les objets NuScenes chargés. À appeler une fois les datasets
    construits et AVANT de créer les DataLoader (num_workers > 0).
    """
    _NUSC_CACHE.clear()
    gc.collect()


def _pose(record):
    return Quaternion(record["rotation"]).rotation_matrix, np.asarray(record["translation"], dtype=np.float64)


def build_sample_info(nusc, sample: Dict, cameras: Sequence[str], classes: Sequence[str], use_lidar: bool):
    """
    Extrait d'un sample nuScenes tout ce dont le dataset a besoin.
    Retourne None si un fichier requis manque sur le disque.
    """
    paths = {}
    for ch in list(cameras) + (["LIDAR_TOP"] if use_lidar else []):
        path = nusc.get_sample_data_path(sample["data"][ch])
        if C.REQUIRE_AVAILABLE_FILES and not os.path.exists(path):
            return None
        paths[ch] = path

    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ref_R, ref_t = _pose(nusc.get("ego_pose", lidar_sd["ego_pose_token"]))
    lidar_R, lidar_t = _pose(nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"]))

    cams = []
    for ch in cameras:
        sd = nusc.get("sample_data", sample["data"][ch])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        cs_R, cs_t = _pose(cs)
        ego_R, ego_t = _pose(nusc.get("ego_pose", sd["ego_pose_token"]))
        cams.append({
            "path": paths[ch], "K": np.asarray(cs["camera_intrinsic"], dtype=np.float64),
            "cs_R": cs_R, "cs_t": cs_t, "ego_R": ego_R, "ego_t": ego_t,
        })

    names, rows = [], []
    for token in sample["anns"]:
        ann = nusc.get("sample_annotation", token)
        name = map_category(ann["category_name"])
        if name is None or name not in classes:
            continue
        vis = ann.get("visibility_token", "4")
        names.append(name)
        # translation (3) | rotation quaternion w,x,y,z (4) | size w,l,h (3) | nb points | visibilité
        rows.append(list(ann["translation"]) + list(ann["rotation"]) + list(ann["size"]) + [
            int(ann.get("num_lidar_pts", 1)) + int(ann.get("num_radar_pts", 0)),
            int(vis) if str(vis).isdigit() else 4,
        ])

    return {
        "token": sample["token"],
        "scene_token": sample["scene_token"],
        "ref_R": ref_R, "ref_t": ref_t,
        "lidar_path": paths.get("LIDAR_TOP"), "lidar_R": lidar_R, "lidar_t": lidar_t,
        "cams": cams,
        "ann_names": names,
        "ann": np.asarray(rows, dtype=np.float64).reshape(-1, 12),
    }


class NuScenesLSSDataset(Dataset):
    def __init__(
        self,
        split: str,
        training: bool,
        dataroot: str = C.DATAROOT,
        version: str = C.VERSION,
        cameras: Sequence[str] = C.CAMERAS,
        classes: Sequence[str] = C.CLASSES,
        use_lidar_depth: bool = True,
        nusc=None,
        verbose: bool = False,
    ):
        super().__init__()
        self.training = bool(training)
        self.cameras = list(cameras)
        self.classes = list(classes)
        self.class_to_idx = {n: i for i, n in enumerate(self.classes)}
        self.use_lidar_depth = bool(use_lidar_depth)

        self.image_aug = LSSImageAugmentation(
            final_dim=C.IMAGE_SIZE,
            training=self.training,
            aug=C.IMG_AUG,
            val_bot_pct=C.VAL_BOT_PCT,
        )

        if nusc is None:
            nusc = get_nuscenes(version, dataroot, verbose)

        from nuscenes.utils.splits import create_splits_scenes
        if version == "v1.0-mini" and split in ("train", "val"):
            split = "mini_" + split
        split_scenes = set(create_splits_scenes()[split])

        candidates = [
            s for s in nusc.sample
            if nusc.get("scene", s["scene_token"])["name"] in split_scenes
        ]
        infos = [build_sample_info(nusc, s, self.cameras, self.classes, True) for s in candidates]
        self.infos = [i for i in infos if i is not None]
        n_scenes = len({i["scene_token"] for i in self.infos})
        n_all_scenes = len({s["scene_token"] for s in candidates})
        print(f"[NuScenesLSSDataset] {version} / {split} : {len(self.infos)}/{len(candidates)} frames "
              f"disponibles sur disque, {n_scenes}/{n_all_scenes} scènes")
        if not self.infos:
            raise RuntimeError(
                f"Aucune frame utilisable pour split={split}, version={version}. Vérifie DATAROOT "
                f"({dataroot}) et que les archives (samples/CAM_*, samples/LIDAR_TOP) y sont extraites."
            )

        self.disable_augmentation = False

    def __len__(self):
        return len(self.infos)

    # ==============================================================
    # Poses (à partir des infos compactes)
    # ==============================================================

    @staticmethod
    def get_reference_pose(info: Dict):
        """Ego au timestamp LIDAR_TOP = repère de référence de la frame."""
        return info["ref_R"], info["ref_t"]

    @staticmethod
    def get_camera_to_reference(info: Dict, cam_index: int):
        """caméra -> ego(t_cam) -> global -> ego de référence."""
        c = info["cams"][cam_index]
        R_glb_ref = info["ref_R"].T
        R_cam_ref = R_glb_ref @ c["ego_R"] @ c["cs_R"]
        t_cam_ref = R_glb_ref @ (c["ego_R"] @ c["cs_t"] + c["ego_t"] - info["ref_t"])
        return R_cam_ref, t_cam_ref, c["K"]

    @staticmethod
    def get_lidar_points_reference(info: Dict) -> np.ndarray:
        """Nuage LIDAR_TOP de la keyframe, exprimé dans l'ego de référence."""
        from nuscenes.utils.data_classes import LidarPointCloud
        pc = LidarPointCloud.from_file(info["lidar_path"])
        # Le lidar est capturé au timestamp de référence : lidar -> ego suffit.
        return transform_points(pc.points[:3].T.astype(np.float64), info["lidar_R"], info["lidar_t"])

    # ==============================================================
    # Annotations
    # ==============================================================

    def get_annotations(self, info: Dict):
        """Annotations des classes suivies, avec les champs utiles au filtrage ET à l'audit."""
        R_glb_ref = info["ref_R"].T
        out = []
        for name, row in zip(info["ann_names"], info["ann"]):
            center = R_glb_ref @ (row[0:3] - info["ref_t"])
            R_box = R_glb_ref @ Quaternion(row[3:7]).rotation_matrix
            yaw = math.atan2(R_box[1, 0], R_box[0, 0])
            w, l, h = (float(v) for v in row[7:10])   # nuScenes : [w, l, h]
            out.append({
                "name": name,
                "label": self.class_to_idx[name],
                "box": [float(center[0]), float(center[1]), float(center[2]), l, w, h, yaw],
                "num_pts": int(row[10]),
                "visibility": int(row[11]),
                "distance": float(math.hypot(center[0], center[1])),
            })
        return out

    @staticmethod
    def keep_annotation(a: Dict) -> bool:
        if C.FILTER_EMPTY_BOXES and a["num_pts"] <= 0:
            return False
        if a["visibility"] < C.MIN_VISIBILITY_LEVEL:
            return False
        if a["distance"] > C.CLASS_RANGE[a["name"]]:
            return False
        return True

    def get_gt(self, info: Dict):
        anns = [a for a in self.get_annotations(info) if self.keep_annotation(a)]
        if not anns:
            return torch.zeros((0, 7), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
        boxes = torch.tensor([a["box"] for a in anns], dtype=torch.float32)
        labels = torch.tensor([a["label"] for a in anns], dtype=torch.long)
        return boxes, labels

    # ==============================================================
    # __getitem__
    # ==============================================================

    def __getitem__(self, index: int):
        info = self.infos[index]
        augment = self.training and not self.disable_augmentation

        # Graine dérivée du RNG torch : DataLoader la rend différente par
        # worker et par itération, et reproductible avec torch.manual_seed.
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        self.image_aug.training = augment

        # Miroir cohérent : un seul tirage pour les 6 images ET le monde BEV.
        mirror = None
        if augment and C.MIRROR_PROB is not None:
            mirror = bool(rng.random() < C.MIRROR_PROB)

        lidar_ref = self.get_lidar_points_reference(info) if self.use_lidar_depth else None

        images, intrins, rots, trans, post_rots, post_trans, depth_bins = [], [], [], [], [], [], []

        for k in range(len(self.cameras)):
            R_cam_ref, t_cam_ref, K = self.get_camera_to_reference(info, k)

            image = Image.open(info["cams"][k]["path"])
            img, post_rot, post_tran = self.image_aug(image, rng, force_flip=mirror)

            images.append(img)
            intrins.append(torch.from_numpy(K).float())
            rots.append(torch.from_numpy(R_cam_ref).float())
            trans.append(torch.from_numpy(t_cam_ref).float())
            post_rots.append(post_rot)
            post_trans.append(post_tran)

            if lidar_ref is not None:
                # ego de référence -> caméra : inverse de (R_cam_ref, t_cam_ref)
                pts_cam = transform_points(lidar_ref, R_cam_ref.T, -R_cam_ref.T @ t_cam_ref)
                bins, _ = build_depth_target(
                    pts_cam, K, post_rot.numpy().astype(np.float64), post_tran.numpy().astype(np.float64),
                    C.IMAGE_SIZE, C.DOWNSAMPLE, C.DEPTH_BOUND,
                )
                depth_bins.append(torch.from_numpy(bins))
            else:
                depth_bins.append(torch.full(C.FEATURE_SIZE, -1, dtype=torch.long))

        gt_boxes, gt_labels = self.get_gt(info)
        bda, _ = sample_bda(rng, C.BDA, training=augment, mirror=mirror)
        gt_boxes = apply_bda_to_boxes(gt_boxes, bda)

        return {
            "images": torch.stack(images),
            "intrins": torch.stack(intrins),
            "rots": torch.stack(rots),
            "trans": torch.stack(trans),
            "post_rots": torch.stack(post_rots),
            "post_trans": torch.stack(post_trans),
            "bda": bda,
            "depth_bins": torch.stack(depth_bins),
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "sample_token": info["token"],
        }
