#!/usr/bin/env python3
"""
Faux nuScenes minimal pour tester le dataset de bout en bout sans
télécharger les données : 6 caméras avec calibrations réalistes,
mouvement ego entre timestamps lidar et caméra, nuage lidar et
annotations construits à partir d'une scène connue en repère ego.
"""

import math
import os

import numpy as np
from PIL import Image
from pyquaternion import Quaternion

CAM_YAWS = {
    "CAM_FRONT": 0.0, "CAM_FRONT_RIGHT": -55.0, "CAM_BACK_RIGHT": -110.0,
    "CAM_BACK": 180.0, "CAM_BACK_LEFT": 110.0, "CAM_FRONT_LEFT": 55.0,
}
OPTICAL_TO_EGO = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])


def rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


class FakeNuScenes:
    def __init__(self, root, scene_name="scene-0061", wall_x=15.0, annotations=()):
        os.makedirs(root, exist_ok=True)
        self.root = root
        self.tables = {k: {} for k in ("scene", "sample_data", "calibrated_sensor", "ego_pose", "sample_annotation")}
        self.paths = {}

        self.tables["scene"]["sc0"] = {"token": "sc0", "name": scene_name}

        # Pose ego de référence (timestamp lidar)
        self.R_ref = rz(30.0)
        self.t_ref = np.array([400.0, 1100.0, 0.0])
        self.tables["ego_pose"]["ep_lidar"] = self._pose(self.R_ref, self.t_ref)

        data = {}
        # --- caméras : ego légèrement avancé de 0.3 m (mouvement entre timestamps)
        rng = np.random.default_rng(0)
        for cam, yaw in CAM_YAWS.items():
            R = rz(yaw) @ OPTICAL_TO_EGO
            t = rz(yaw) @ np.array([1.5, 0.0, 1.55])
            f = 809.0 if cam == "CAM_BACK" else 1266.0
            K = [[f, 0, 816.0], [0, f, 491.0], [0, 0, 1.0]]
            self.tables["calibrated_sensor"][f"cs_{cam}"] = {
                "rotation": list(Quaternion(matrix=R).elements), "translation": list(t), "camera_intrinsic": K,
            }
            R_ego = self.R_ref
            t_ego = self.t_ref + self.R_ref @ np.array([0.3, 0.0, 0.0])
            self.tables["ego_pose"][f"ep_{cam}"] = self._pose(R_ego, t_ego)
            self.tables["sample_data"][f"sd_{cam}"] = {
                "calibrated_sensor_token": f"cs_{cam}", "ego_pose_token": f"ep_{cam}",
            }
            path = os.path.join(root, f"{cam}.jpg")
            Image.fromarray(rng.integers(0, 255, (900, 1600, 3), dtype=np.uint8)).save(path, quality=50)
            self.paths[f"sd_{cam}"] = path
            data[cam] = f"sd_{cam}"

        # --- lidar : rotation -90° autour de z, comme nuScenes
        R_l, t_l = rz(-90.0), np.array([0.94, 0.0, 1.84])
        self.tables["calibrated_sensor"]["cs_lidar"] = {
            "rotation": list(Quaternion(matrix=R_l).elements), "translation": list(t_l),
        }
        self.tables["sample_data"]["sd_lidar"] = {"calibrated_sensor_token": "cs_lidar", "ego_pose_token": "ep_lidar"}

        # Mur vertical à x = wall_x (repère ego de référence), devant.
        ys, zs = np.meshgrid(np.linspace(-6, 6, 121), np.linspace(-1.0, 2.5, 36))
        wall = np.stack([np.full(ys.size, wall_x), ys.ravel(), zs.ravel()], 1)
        self.wall_points_ego = wall
        pts_lidar = (wall - t_l) @ R_l          # = R_l^T (p - t)
        arr = np.zeros((pts_lidar.shape[0], 5), dtype=np.float32)
        arr[:, :3] = pts_lidar
        lidar_path = os.path.join(root, "lidar.pcd.bin")
        arr.tofile(lidar_path)
        self.paths["sd_lidar"] = lidar_path
        data["LIDAR_TOP"] = "sd_lidar"

        # --- annotations : (catégorie, centre ego, yaw, taille [w,l,h], nb pts)
        anns = []
        for i, (cat, center, yaw, size, npts) in enumerate(annotations):
            c_glb = self.R_ref @ np.asarray(center, dtype=float) + self.t_ref
            q = Quaternion(matrix=self.R_ref @ rz(math.degrees(yaw)))
            tok = f"ann{i}"
            self.tables["sample_annotation"][tok] = {
                "category_name": cat, "translation": list(c_glb), "rotation": list(q.elements),
                "size": list(size), "num_lidar_pts": npts, "num_radar_pts": 0, "visibility_token": "4",
            }
            anns.append(tok)

        self.sample = [{"token": "s0", "scene_token": "sc0", "data": data, "anns": anns}]

    @staticmethod
    def _pose(R, t):
        return {"rotation": list(Quaternion(matrix=R).elements), "translation": list(t)}

    def get(self, table, token):
        return self.tables[table][token]

    def get_sample_data_path(self, token):
        return self.paths[token]
