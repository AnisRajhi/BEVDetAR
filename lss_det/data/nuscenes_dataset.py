#!/usr/bin/env python3

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import (
    create_splits_scenes,
)

from lss_det.data.transforms import (
    LSSImageTransform,
)


DEFAULT_CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]


DEFAULT_CLASSES = [
    "car",
    "truck",
    "bus",
    "pedestrian",
    "bicycle",
    "motorcycle",
]


class NuScenesLSSDataset(Dataset):
    """
    nuScenes adapter pour notre LSSDetector.

    ==============================================================
    OUTPUT sample
    ==============================================================

    images:
        [N,3,H,W]

    intrins:
        [N,3,3]

    rots:
        [N,3,3]

        camera -> reference ego

    trans:
        [N,3]

    post_rots:
        [N,3,3]

    post_trans:
        [N,3]

    gt_boxes:
        [M,7]

        [x,y,z,l,w,h,yaw]

    gt_labels:
        [M]
    """

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-mini",
        split: str = "train",
        cameras: Sequence[str] = DEFAULT_CAMERAS,
        classes: Sequence[str] = DEFAULT_CLASSES,
        image_transform: Optional[
            LSSImageTransform
        ] = None,
        verbose: bool = False,
    ):
        super().__init__()

        self.dataroot = dataroot
        self.version = version

        self.cameras = list(
            cameras
        )

        self.classes = list(
            classes
        )

        self.class_to_idx = {
            name: i
            for i, name
            in enumerate(
                self.classes
            )
        }

        if image_transform is None:

            image_transform = (
                LSSImageTransform(
                    final_dim=(
                        128,
                        352,
                    ),
                    bottom_crop_pct=0.11,
                    normalize=True,
                )
            )

        self.image_transform = (
            image_transform
        )

        # ==========================================================
        # nuScenes API
        # ==========================================================

        self.nusc = NuScenes(
            version=self.version,
            dataroot=self.dataroot,
            verbose=verbose,
        )

        # ==========================================================
        # Split
        # ==========================================================

        self.split = (
            self._resolve_split(
                split
            )
        )

        split_scenes = set(
            create_splits_scenes()[
                self.split
            ]
        )

        # Garder uniquement les samples
        # appartenant aux scènes du split.

        self.samples = []

        for sample in self.nusc.sample:

            scene = self.nusc.get(
                "scene",
                sample[
                    "scene_token"
                ],
            )

            if (
                scene["name"]
                in split_scenes
            ):

                self.samples.append(
                    sample
                )

        if len(self.samples) == 0:

            raise RuntimeError(
                f"No samples found for "
                f"split={self.split}, "
                f"version={self.version}"
            )

    # ==============================================================
    # Split mapping
    # ==============================================================

    def _resolve_split(
        self,
        split: str,
    ) -> str:

        # On autorise:
        #
        # split="train"
        #
        # même avec v1.0-mini.

        if self.version == "v1.0-mini":

            if split == "train":
                return "mini_train"

            if split == "val":
                return "mini_val"

        return split

    def __len__(
        self,
    ):

        return len(
            self.samples
        )

    # ==============================================================
    # Category mapping
    # ==============================================================

    def _map_category(
        self,
        category_name: str,
    ) -> Optional[int]:
        """
        nuScenes utilise des catégories détaillées:

            vehicle.car
            vehicle.bus.rigid
            human.pedestrian.adult
            ...

        On les ramène vers nos classes simples.
        """

        mapped = None

        if category_name == "vehicle.car":

            mapped = "car"

        elif category_name == "vehicle.truck":

            mapped = "truck"

        elif category_name.startswith(
            "vehicle.bus."
        ):

            mapped = "bus"

        elif category_name.startswith(
            "human.pedestrian."
        ):

            mapped = "pedestrian"

        elif category_name == (
            "vehicle.bicycle"
        ):

            mapped = "bicycle"

        elif category_name == (
            "vehicle.motorcycle"
        ):

            mapped = "motorcycle"

        if mapped is None:

            return None

        return self.class_to_idx.get(
            mapped,
            None,
        )

    # ==============================================================
    # Reference ego pose
    # ==============================================================

    def _get_reference_pose(
        self,
        sample: Dict,
    ):

        """
        On utilise LIDAR_TOP comme timestamp / ego frame
        de référence pour la sample.
        """

        lidar_token = sample[
            "data"
        ][
            "LIDAR_TOP"
        ]

        lidar_sd = self.nusc.get(
            "sample_data",
            lidar_token,
        )

        ego_pose = self.nusc.get(
            "ego_pose",
            lidar_sd[
                "ego_pose_token"
            ],
        )

        # Ego reference -> global
        R_ref_global = Quaternion(
            ego_pose["rotation"]
        ).rotation_matrix

        t_ref_global = np.asarray(
            ego_pose[
                "translation"
            ],
            dtype=np.float64,
        )

        return (
            R_ref_global,
            t_ref_global,
        )

    # ==============================================================
    # Camera calibration
    # ==============================================================

    def _get_camera_to_reference(
        self,
        camera_token: str,
        R_ref_global: np.ndarray,
        t_ref_global: np.ndarray,
    ):
        """
        Calcule:

            camera optical frame
                ↓
            reference ego

        en tenant compte du ego pose au timestamp caméra.
        """

        sample_data = self.nusc.get(
            "sample_data",
            camera_token,
        )

        calibrated_sensor = (
            self.nusc.get(
                "calibrated_sensor",
                sample_data[
                    "calibrated_sensor_token"
                ],
            )
        )

        camera_ego_pose = (
            self.nusc.get(
                "ego_pose",
                sample_data[
                    "ego_pose_token"
                ],
            )
        )

        # ----------------------------------------------------------
        # Camera -> ego(camera timestamp)
        # ----------------------------------------------------------

        R_cam_ego = Quaternion(
            calibrated_sensor[
                "rotation"
            ]
        ).rotation_matrix

        t_cam_ego = np.asarray(
            calibrated_sensor[
                "translation"
            ],
            dtype=np.float64,
        )

        # ----------------------------------------------------------
        # Ego(camera timestamp) -> global
        # ----------------------------------------------------------

        R_ego_global = Quaternion(
            camera_ego_pose[
                "rotation"
            ]
        ).rotation_matrix

        t_ego_global = np.asarray(
            camera_ego_pose[
                "translation"
            ],
            dtype=np.float64,
        )

        # ----------------------------------------------------------
        # Global -> reference ego
        # ----------------------------------------------------------

        R_global_ref = (
            R_ref_global.T
        )

        # ----------------------------------------------------------
        # Rotation caméra -> reference ego
        # ----------------------------------------------------------

        R_cam_ref = (
            R_global_ref
            @ R_ego_global
            @ R_cam_ego
        )

        # ----------------------------------------------------------
        # Translation caméra -> reference ego
        # ----------------------------------------------------------

        camera_origin_global = (
            R_ego_global
            @ t_cam_ego
            + t_ego_global
        )

        t_cam_ref = (
            R_global_ref
            @ (
                camera_origin_global
                - t_ref_global
            )
        )

        # ----------------------------------------------------------
        # Intrinsics
        # ----------------------------------------------------------

        intrinsic = np.asarray(
            calibrated_sensor[
                "camera_intrinsic"
            ],
            dtype=np.float64,
        )

        return (
            R_cam_ref,
            t_cam_ref,
            intrinsic,
            sample_data,
        )

    # ==============================================================
    # GT boxes
    # ==============================================================

    def _get_gt(
        self,
        sample: Dict,
        R_ref_global: np.ndarray,
        t_ref_global: np.ndarray,
    ):

        boxes = []
        labels = []

        R_global_ref = (
            R_ref_global.T
        )

        for annotation_token in sample[
            "anns"
        ]:

            ann = self.nusc.get(
                "sample_annotation",
                annotation_token,
            )

            class_id = (
                self._map_category(
                    ann[
                        "category_name"
                    ]
                )
            )

            # Classe non utilisée.
            if class_id is None:
                continue

            # ------------------------------------------------------
            # Center global -> reference ego
            # ------------------------------------------------------

            center_global = np.asarray(
                ann[
                    "translation"
                ],
                dtype=np.float64,
            )

            center_ref = (
                R_global_ref
                @ (
                    center_global
                    - t_ref_global
                )
            )

            # ------------------------------------------------------
            # Orientation global -> reference ego
            # ------------------------------------------------------

            R_box_global = Quaternion(
                ann[
                    "rotation"
                ]
            ).rotation_matrix

            R_box_ref = (
                R_global_ref
                @ R_box_global
            )

            # Notre convention:
            #
            # yaw=0:
            # +X forward
            #
            # yaw positif:
            # rotation CCW autour de +Z.

            yaw = math.atan2(
                R_box_ref[
                    1,
                    0,
                ],
                R_box_ref[
                    0,
                    0,
                ],
            )

            # ------------------------------------------------------
            # nuScenes size:
            #
            # [width, length, height]
            #
            # Notre convention:
            #
            # [length, width, height]
            # ------------------------------------------------------

            width = float(
                ann["size"][0]
            )

            length = float(
                ann["size"][1]
            )

            height = float(
                ann["size"][2]
            )

            boxes.append(
                [
                    float(
                        center_ref[0]
                    ),
                    float(
                        center_ref[1]
                    ),
                    float(
                        center_ref[2]
                    ),
                    length,
                    width,
                    height,
                    yaw,
                ]
            )

            labels.append(
                class_id
            )

        if len(boxes) == 0:

            gt_boxes = torch.empty(
                (
                    0,
                    7,
                ),
                dtype=torch.float32,
            )

            gt_labels = torch.empty(
                (
                    0,
                ),
                dtype=torch.long,
            )

        else:

            gt_boxes = torch.tensor(
                boxes,
                dtype=torch.float32,
            )

            gt_labels = torch.tensor(
                labels,
                dtype=torch.long,
            )

        return (
            gt_boxes,
            gt_labels,
        )

    # ==============================================================
    # Get item
    # ==============================================================

    def __getitem__(
        self,
        index: int,
    ):

        sample = self.samples[
            index
        ]

        # ==========================================================
        # Reference frame
        # ==========================================================

        (
            R_ref_global,
            t_ref_global,
        ) = self._get_reference_pose(
            sample
        )

        images = []

        intrins = []

        rots = []

        trans = []

        post_rots = []

        post_trans = []

        # ==========================================================
        # Cameras
        # ==========================================================

        for camera_name in self.cameras:

            camera_token = sample[
                "data"
            ][
                camera_name
            ]

            (
                R_cam_ref,
                t_cam_ref,
                intrinsic,
                sample_data,
            ) = self._get_camera_to_reference(
                camera_token,
                R_ref_global,
                t_ref_global,
            )

            # ------------------------------------------------------
            # Image
            # ------------------------------------------------------

            image_path = (
                self.nusc
                .get_sample_data_path(
                    camera_token
                )
            )

            image = Image.open(
                image_path
            )

            (
                image_tensor,
                post_rot,
                post_tran,
            ) = self.image_transform(
                image
            )

            images.append(
                image_tensor
            )

            intrins.append(
                torch.tensor(
                    intrinsic,
                    dtype=torch.float32,
                )
            )

            rots.append(
                torch.tensor(
                    R_cam_ref,
                    dtype=torch.float32,
                )
            )

            trans.append(
                torch.tensor(
                    t_cam_ref,
                    dtype=torch.float32,
                )
            )

            post_rots.append(
                post_rot
            )

            post_trans.append(
                post_tran
            )

        # ==========================================================
        # Stack cameras
        # ==========================================================

        images = torch.stack(
            images,
            dim=0,
        )

        intrins = torch.stack(
            intrins,
            dim=0,
        )

        rots = torch.stack(
            rots,
            dim=0,
        )

        trans = torch.stack(
            trans,
            dim=0,
        )

        post_rots = torch.stack(
            post_rots,
            dim=0,
        )

        post_trans = torch.stack(
            post_trans,
            dim=0,
        )

        # ==========================================================
        # Ground truth
        # ==========================================================

        (
            gt_boxes,
            gt_labels,
        ) = self._get_gt(
            sample,
            R_ref_global,
            t_ref_global,
        )

        return {
            "images": images,

            "intrins": intrins,

            "rots": rots,

            "trans": trans,

            "post_rots": post_rots,

            "post_trans": post_trans,

            "gt_boxes": gt_boxes,

            "gt_labels": gt_labels,

            "sample_token": sample[
                "token"
            ],
        }