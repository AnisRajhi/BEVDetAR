#!/usr/bin/env python3

from typing import Tuple

import torch
import torch.nn as nn


class Geometry(nn.Module):
    """
    LSS geometry module.

    Ce module construit un frustum fixe (u, v, d), puis convertit
    chaque hypothèse en une coordonnée 3D dans le repère ego/robot.

    ------------------------------------------------------------------
    FRUSTUM
    ------------------------------------------------------------------

    [D, Hf, Wf, 3]

    avec:
        3 = (u, v, d)

    Exemple:
        D  = 41
        Hf = 8
        Wf = 22

        frustum.shape = [41, 8, 22, 3]


    ------------------------------------------------------------------
    INPUTS forward()
    ------------------------------------------------------------------

    intrins:
        [B, N, 3, 3]

    rots:
        [B, N, 3, 3]

        Rotation caméra -> ego/robot.

    trans:
        [B, N, 3]

        Translation caméra -> ego/robot.

    post_rots:
        [B, N, 3, 3]

        Transformations appliquées aux coordonnées image
        lors du preprocessing.

    post_trans:
        [B, N, 3]


    ------------------------------------------------------------------
    OUTPUT
    ------------------------------------------------------------------

    geometry:
        [B, N, D, Hf, Wf, 3]

    avec:
        3 = (X, Y, Z) dans le repère ego/robot.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (128, 352),
        feature_size: Tuple[int, int] = (8, 22),
        depth_bound: Tuple[float, float, float] = (
            4.0,
            45.0,
            1.0,
        ),
    ):
        super().__init__()

        self.image_h = int(image_size[0])
        self.image_w = int(image_size[1])

        self.feature_h = int(feature_size[0])
        self.feature_w = int(feature_size[1])

        self.depth_min = float(depth_bound[0])
        self.depth_max = float(depth_bound[1])
        self.depth_step = float(depth_bound[2])

        # --------------------------------------------------------------
        # Construction du frustum fixe
        # --------------------------------------------------------------

        frustum = self.create_frustum()

        # register_buffer:
        #
        # - ce n'est PAS un paramètre appris
        # - il suit automatiquement model.to("cuda")
        # - il est sauvegardé dans le state_dict
        #
        self.register_buffer(
            "frustum",
            frustum,
            persistent=True,
        )

    @property
    def num_depth_bins(self) -> int:
        """
        Nombre de depth bins D.
        """

        return int(
            (self.depth_max - self.depth_min)
            / self.depth_step
        )

    def create_frustum(self) -> torch.Tensor:
        """
        Construit le frustum fixe.

        OUTPUT
        ------
        frustum:
            [D, Hf, Wf, 3]

        où:
            frustum[d, v, u] = (u_image, v_image, depth)
        """

        # --------------------------------------------------------------
        # 1. Depth candidates
        # --------------------------------------------------------------
        #
        # Exemple:
        #
        # [4, 5, 6, ..., 44]
        #
        depths = torch.arange(
            self.depth_min,
            self.depth_max,
            self.depth_step,
            dtype=torch.float32,
        )

        D = depths.shape[0]

        # [D]
        #
        #   ↓
        #
        # [D, 1, 1]
        #
        #   ↓ expand
        #
        # [D, Hf, Wf]
        #
        ds = depths.view(
            D,
            1,
            1,
        ).expand(
            D,
            self.feature_h,
            self.feature_w,
        )

        # --------------------------------------------------------------
        # 2. Horizontal image coordinates
        # --------------------------------------------------------------
        #
        # On veut Wf positions réparties sur la largeur de l'image.
        #
        # Exemple:
        #
        # image width = 352
        # feature width = 22
        #
        # xs contient 22 positions entre 0 et 351.
        #
        xs = torch.linspace(
            0.0,
            float(self.image_w - 1),
            self.feature_w,
            dtype=torch.float32,
        )

        # [Wf]
        #
        # ↓
        #
        # [1, 1, Wf]
        #
        # ↓
        #
        # [D, Hf, Wf]
        #
        xs = xs.view(
            1,
            1,
            self.feature_w,
        ).expand(
            D,
            self.feature_h,
            self.feature_w,
        )

        # --------------------------------------------------------------
        # 3. Vertical image coordinates
        # --------------------------------------------------------------

        ys = torch.linspace(
            0.0,
            float(self.image_h - 1),
            self.feature_h,
            dtype=torch.float32,
        )

        ys = ys.view(
            1,
            self.feature_h,
            1,
        ).expand(
            D,
            self.feature_h,
            self.feature_w,
        )

        # --------------------------------------------------------------
        # 4. Stack
        # --------------------------------------------------------------
        #
        # xs : [D,Hf,Wf]
        # ys : [D,Hf,Wf]
        # ds : [D,Hf,Wf]
        #
        # ↓
        #
        # [D,Hf,Wf,3]
        #
        # 3 = (u,v,d)
        #
        frustum = torch.stack(
            (
                xs,
                ys,
                ds,
            ),
            dim=-1,
        )

        return frustum

    @staticmethod
    def _check_matrix_shape(
        tensor: torch.Tensor,
        expected_last_dims,
        name: str,
    ):
        if tensor.shape[-len(expected_last_dims):] != expected_last_dims:
            raise ValueError(
                f"{name}: unexpected shape "
                f"{tuple(tensor.shape)}. "
                f"Expected last dimensions "
                f"{expected_last_dims}."
            )

    def forward(
        self,
        intrins: torch.Tensor,
        rots: torch.Tensor,
        trans: torch.Tensor,
        post_rots: torch.Tensor,
        post_trans: torch.Tensor,
    ) -> torch.Tensor:

        # --------------------------------------------------------------
        # 0. Shape checks
        # --------------------------------------------------------------

        self._check_matrix_shape(
            intrins,
            (3, 3),
            "intrins",
        )

        self._check_matrix_shape(
            rots,
            (3, 3),
            "rots",
        )

        self._check_matrix_shape(
            post_rots,
            (3, 3),
            "post_rots",
        )

        if trans.ndim != 3 or trans.shape[-1] != 3:
            raise ValueError(
                "trans must have shape [B,N,3]. "
                f"Received {tuple(trans.shape)}"
            )

        if (
            post_trans.ndim != 3
            or post_trans.shape[-1] != 3
        ):
            raise ValueError(
                "post_trans must have shape [B,N,3]. "
                f"Received {tuple(post_trans.shape)}"
            )

        B, N = intrins.shape[:2]

        if rots.shape[:2] != (B, N):
            raise ValueError(
                "rots has different B/N dimensions."
            )

        if trans.shape[:2] != (B, N):
            raise ValueError(
                "trans has different B/N dimensions."
            )

        if post_rots.shape[:2] != (B, N):
            raise ValueError(
                "post_rots has different B/N dimensions."
            )

        if post_trans.shape[:2] != (B, N):
            raise ValueError(
                "post_trans has different B/N dimensions."
            )

        D = self.num_depth_bins
        Hf = self.feature_h
        Wf = self.feature_w

        # --------------------------------------------------------------
        # 1. Dupliquer virtuellement le frustum pour B et N
        # --------------------------------------------------------------
        #
        # self.frustum:
        #
        # [D,Hf,Wf,3]
        #
        # ↓
        #
        # [1,1,D,Hf,Wf,3]
        #
        # ↓ expand
        #
        # [B,N,D,Hf,Wf,3]
        #
        points = self.frustum.to(
            dtype=intrins.dtype,
            device=intrins.device,
        )

        points = points.view(
            1,
            1,
            D,
            Hf,
            Wf,
            3,
        ).expand(
            B,
            N,
            D,
            Hf,
            Wf,
            3,
        )

        # --------------------------------------------------------------
        # 2. Undo post_trans
        # --------------------------------------------------------------
        #
        # p_aug = Rpost * p_original + tpost
        #
        # Donc:
        #
        # p_original =
        #     Rpost^-1 * (p_aug - tpost)
        #
        points = points - post_trans.view(
            B,
            N,
            1,
            1,
            1,
            3,
        )

        # --------------------------------------------------------------
        # 3. Undo post_rots
        # --------------------------------------------------------------

        inv_post_rots = torch.linalg.inv(
            post_rots
        )

        # points:
        #
        # [B,N,D,Hf,Wf,3]
        #
        # ↓ unsqueeze
        #
        # [B,N,D,Hf,Wf,3,1]
        #
        points = torch.matmul(
            inv_post_rots.view(
                B,
                N,
                1,
                1,
                1,
                3,
                3,
            ),
            points.unsqueeze(-1),
        ).squeeze(-1)

        # Ici points contient de nouveau:
        #
        # (u_original, v_original, d)

        # --------------------------------------------------------------
        # 4. Transformer (u,v,d) en (u*d, v*d, d)
        # --------------------------------------------------------------
        #
        # Pour la rétroprojection pinhole:
        #
        # P_camera =
        #
        # K^-1 @
        #
        # [u*d]
        # [v*d]
        # [ d ]
        #
        uv = points[..., 0:2]
        depth = points[..., 2:3]

        points = torch.cat(
            (
                uv * depth,
                depth,
            ),
            dim=-1,
        )

        # Maintenant:
        #
        # points = (u*d, v*d, d)

        # --------------------------------------------------------------
        # 5. Pixel -> camera XYZ
        #
        # puis camera -> robot XYZ
        # --------------------------------------------------------------
        #
        # Pcam =
        #
        # K^-1 @ [u*d, v*d, d]
        #
        #
        # Probot =
        #
        # R @ Pcam + t
        #
        #
        # donc:
        #
        # Probot =
        #
        # R @ K^-1 @ pixel_depth + t
        #
        inv_intrins = torch.linalg.inv(
            intrins
        )

        camera_to_robot_projection = torch.matmul(
            rots,
            inv_intrins,
        )

        # Shape:
        #
        # [B,N,3,3]
        #
        # ↓
        #
        # [B,N,1,1,1,3,3]
        #
        geometry = torch.matmul(
            camera_to_robot_projection.view(
                B,
                N,
                1,
                1,
                1,
                3,
                3,
            ),
            points.unsqueeze(-1),
        ).squeeze(-1)

        # --------------------------------------------------------------
        # 6. Ajouter la translation caméra -> robot
        # --------------------------------------------------------------

        geometry = geometry + trans.view(
            B,
            N,
            1,
            1,
            1,
            3,
        )

        # --------------------------------------------------------------
        # OUTPUT
        # --------------------------------------------------------------
        #
        # [B,N,D,Hf,Wf,3]
        #
        # 3 = (X,Y,Z) robot/ego
        #
        return geometry


if __name__ == "__main__":

    geometry_module = Geometry(
        image_size=(128, 352),
        feature_size=(8, 22),
        depth_bound=(4.0, 45.0, 1.0),
    )

    B = 2
    N = 4

    # Calibration artificielle simple.
    intrins = torch.eye(3).view(
        1, 1, 3, 3
    ).repeat(
        B, N, 1, 1
    )

    rots = torch.eye(3).view(
        1, 1, 3, 3
    ).repeat(
        B, N, 1, 1
    )

    trans = torch.zeros(
        B,
        N,
        3,
    )

    post_rots = torch.eye(3).view(
        1, 1, 3, 3
    ).repeat(
        B, N, 1, 1
    )

    post_trans = torch.zeros(
        B,
        N,
        3,
    )

    xyz = geometry_module(
        intrins,
        rots,
        trans,
        post_rots,
        post_trans,
    )

    print("Frustum:", geometry_module.frustum.shape)

    print("Geometry:", xyz.shape)

    assert geometry_module.frustum.shape == (
        41,
        8,
        22,
        3,
    )

    assert xyz.shape == (
        B,
        N,
        41,
        8,
        22,
        3,
    )

    print("Geometry basic test PASSED")