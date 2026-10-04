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
            from nuscenes.nuscenes import NuScenes
            nusc = NuScenes(version=version, dataroot=os.path.expanduser(dataroot), verbose=verbose)
        self.nusc = nusc

        from nuscenes.utils.splits import create_splits_scenes
        if version == "v1.0-mini" and split in ("train", "val"):
            split = "mini_" + split
        split_scenes = set(create_splits_scenes()[split])

        self.samples = [
            s for s in self.nusc.sample
            if self.nusc.get("scene", s["scene_token"])["name"] in split_scenes
        ]
        if not self.samples:
            raise RuntimeError(f"Aucun sample pour split={split}, version={version}")

        self.disable_augmentation = False

    def __len__(self):
        return len(self.samples)

    # ==============================================================
    # Poses
    # ==============================================================

    def get_reference_pose(self, sample: Dict):
        """Ego au timestamp LIDAR_TOP = repère de référence de la frame."""
        lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        return Quaternion(ego["rotation"]).rotation_matrix, np.asarray(ego["translation"], dtype=np.float64)

    def get_camera_to_reference(self, camera_token: str, R_ref_global, t_ref_global):
        """caméra -> ego(t_cam) -> global -> ego de référence."""
        sd = self.nusc.get("sample_data", camera_token)
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ego = self.nusc.get("ego_pose", sd["ego_pose_token"])

        R_cam_ego = Quaternion(cs["rotation"]).rotation_matrix
        t_cam_ego = np.asarray(cs["translation"], dtype=np.float64)
        R_ego_glb = Quaternion(ego["rotation"]).rotation_matrix
        t_ego_glb = np.asarray(ego["translation"], dtype=np.float64)

        R_glb_ref = R_ref_global.T
        R_cam_ref = R_glb_ref @ R_ego_glb @ R_cam_ego
        t_cam_ref = R_glb_ref @ (R_ego_glb @ t_cam_ego + t_ego_glb - t_ref_global)
        K = np.asarray(cs["camera_intrinsic"], dtype=np.float64)
        return R_cam_ref, t_cam_ref, K

    def get_lidar_points_reference(self, sample: Dict) -> np.ndarray:
        """Nuage LIDAR_TOP de la keyframe, exprimé dans l'ego de référence."""
        from nuscenes.utils.data_classes import LidarPointCloud

        token = sample["data"]["LIDAR_TOP"]
        sd = self.nusc.get("sample_data", token)
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        pc = LidarPointCloud.from_file(self.nusc.get_sample_data_path(token))
        pts = pc.points[:3].T.astype(np.float64)
        # Le lidar est capturé au timestamp de référence : lidar -> ego suffit.
        return transform_points(
            pts, Quaternion(cs["rotation"]).rotation_matrix, np.asarray(cs["translation"], dtype=np.float64)
        )

    # ==============================================================
    # Annotations
    # ==============================================================

    def get_annotations(self, sample: Dict, R_ref_global, t_ref_global):
        """
        Retourne toutes les annotations des classes suivies, avec les
        champs utiles au filtrage ET à l'audit.
        """
        R_glb_ref = R_ref_global.T
        out = []
        for token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", token)
            name = map_category(ann["category_name"])
            if name is None or name not in self.class_to_idx:
                continue

            center = R_glb_ref @ (np.asarray(ann["translation"], dtype=np.float64) - t_ref_global)
            R_box = R_glb_ref @ Quaternion(ann["rotation"]).rotation_matrix
            yaw = math.atan2(R_box[1, 0], R_box[0, 0])
            w, l, h = (float(v) for v in ann["size"])   # nuScenes : [w, l, h]

            vis = ann.get("visibility_token", "4")
            out.append({
                "name": name,
                "label": self.class_to_idx[name],
                "box": [float(center[0]), float(center[1]), float(center[2]), l, w, h, yaw],
                "num_pts": int(ann.get("num_lidar_pts", 1)) + int(ann.get("num_radar_pts", 0)),
                "visibility": int(vis) if str(vis).isdigit() else 4,
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

    def get_gt(self, sample: Dict, R_ref_global, t_ref_global):
        anns = [a for a in self.get_annotations(sample, R_ref_global, t_ref_global) if self.keep_annotation(a)]
        if not anns:
            return torch.zeros((0, 7), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
        boxes = torch.tensor([a["box"] for a in anns], dtype=torch.float32)
        labels = torch.tensor([a["label"] for a in anns], dtype=torch.long)
        return boxes, labels

    # ==============================================================
    # __getitem__
    # ==============================================================

    def __getitem__(self, index: int):
        sample = self.samples[index]
        augment = self.training and not self.disable_augmentation

        # Graine dérivée du RNG torch : DataLoader la rend différente par
        # worker et par itération, et reproductible avec torch.manual_seed.
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        self.image_aug.training = augment

        R_ref_global, t_ref_global = self.get_reference_pose(sample)
        lidar_ref = self.get_lidar_points_reference(sample) if self.use_lidar_depth else None

        images, intrins, rots, trans, post_rots, post_trans, depth_bins = [], [], [], [], [], [], []

        for cam in self.cameras:
            token = sample["data"][cam]
            R_cam_ref, t_cam_ref, K = self.get_camera_to_reference(token, R_ref_global, t_ref_global)

            image = Image.open(self.nusc.get_sample_data_path(token))
            img, post_rot, post_tran = self.image_aug(image, rng)

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

        gt_boxes, gt_labels = self.get_gt(sample, R_ref_global, t_ref_global)
        bda, _ = sample_bda(rng, C.BDA, training=augment)
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
            "sample_token": sample["token"],
        }
