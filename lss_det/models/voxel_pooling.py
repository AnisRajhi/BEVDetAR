#!/usr/bin/env python3

from typing import Tuple

import torch
import torch.nn as nn


class VoxelPooling(nn.Module):
    """
    Projection / agrégation des features LSS dans une grille BEV.

    Ce module ne contient AUCUN poids appris.

    Il fait:

        XYZ continu
            ↓
        voxel indices (ix, iy, iz)
            ↓
        suppression des points hors grille
            ↓
        somme des features appartenant au même voxel
            ↓
        tenseur BEV


    ------------------------------------------------------------------
    INPUTS
    ------------------------------------------------------------------

    features:
        [B, N, D, Hf, Wf, C]

    geometry:
        [B, N, D, Hf, Wf, 3]

        3 = (X, Y, Z)


    ------------------------------------------------------------------
    OUTPUT
    ------------------------------------------------------------------

    Si Nz == 1:

        [B, C, Nx, Ny]

    Si Nz > 1 et collapse_z == True:

        [B, C*Nz, Nx, Ny]


    Exemple actuel:

        xbound = [-50, 50, 0.5]
        ybound = [-50, 50, 0.5]
        zbound = [-10, 10, 20]

        Nx = 200
        Ny = 200
        Nz = 1

        output:
        [B, 64, 200, 200]
    """

    def __init__(
        self,
        xbound: Tuple[float, float, float] = (
            -50.0,
            50.0,
            0.5,
        ),
        ybound: Tuple[float, float, float] = (
            -50.0,
            50.0,
            0.5,
        ),
        zbound: Tuple[float, float, float] = (
            -10.0,
            10.0,
            20.0,
        ),
        collapse_z: bool = True,
    ):
        super().__init__()

        self.xbound = tuple(float(v) for v in xbound)
        self.ybound = tuple(float(v) for v in ybound)
        self.zbound = tuple(float(v) for v in zbound)

        self.collapse_z = collapse_z

        # --------------------------------------------------------------
        # Bornes minimales
        # --------------------------------------------------------------

        mins = torch.tensor(
            [
                self.xbound[0],
                self.ybound[0],
                self.zbound[0],
            ],
            dtype=torch.float32,
        )

        # --------------------------------------------------------------
        # Résolution des voxels
        # --------------------------------------------------------------

        voxel_size = torch.tensor(
            [
                self.xbound[2],
                self.ybound[2],
                self.zbound[2],
            ],
            dtype=torch.float32,
        )

        # --------------------------------------------------------------
        # Nombre de voxels
        # --------------------------------------------------------------

        grid_size = torch.tensor(
            [
                int(
                    round(
                        (
                            self.xbound[1]
                            - self.xbound[0]
                        )
                        / self.xbound[2]
                    )
                ),
                int(
                    round(
                        (
                            self.ybound[1]
                            - self.ybound[0]
                        )
                        / self.ybound[2]
                    )
                ),
                int(
                    round(
                        (
                            self.zbound[1]
                            - self.zbound[0]
                        )
                        / self.zbound[2]
                    )
                ),
            ],
            dtype=torch.long,
        )

        # Ces valeurs ne sont pas apprises.
        # On les enregistre comme buffers pour qu'elles suivent
        # automatiquement CPU / CUDA.

        self.register_buffer(
            "grid_min",
            mins,
            persistent=True,
        )

        self.register_buffer(
            "voxel_size",
            voxel_size,
            persistent=True,
        )

        self.register_buffer(
            "grid_size",
            grid_size,
            persistent=True,
        )

    @property
    def Nx(self) -> int:
        return int(self.grid_size[0].item())

    @property
    def Ny(self) -> int:
        return int(self.grid_size[1].item())

    @property
    def Nz(self) -> int:
        return int(self.grid_size[2].item())

    def forward(
        self,
        features: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:

        # ==============================================================
        # 0. Vérifications
        # ==============================================================

        if features.ndim != 6:
            raise ValueError(
                "features must have shape "
                "[B,N,D,Hf,Wf,C]. "
                f"Received {tuple(features.shape)}"
            )

        if geometry.ndim != 6:
            raise ValueError(
                "geometry must have shape "
                "[B,N,D,Hf,Wf,3]. "
                f"Received {tuple(geometry.shape)}"
            )

        if geometry.shape[-1] != 3:
            raise ValueError(
                "geometry last dimension must be XYZ = 3. "
                f"Received {geometry.shape[-1]}"
            )

        # Les cinq premières dimensions doivent être identiques.
        if features.shape[:5] != geometry.shape[:5]:
            raise ValueError(
                "features and geometry are not aligned:\n"
                f"features: {tuple(features.shape)}\n"
                f"geometry: {tuple(geometry.shape)}"
            )

        B, N, D, Hf, Wf, C = features.shape

        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz

        # ==============================================================
        # 1. Flatten de tous les points
        # ==============================================================
        #
        # Avant:
        #
        # features:
        # [B,N,D,Hf,Wf,C]
        #
        # geometry:
        # [B,N,D,Hf,Wf,3]
        #
        # Après:
        #
        # features_flat:
        # [P,C]
        #
        # geometry_flat:
        # [P,3]
        #
        # avec:
        #
        # P = B*N*D*Hf*Wf
        #

        features_flat = features.reshape(
            -1,
            C,
        )

        geometry_flat = geometry.reshape(
            -1,
            3,
        )

        # ==============================================================
        # 2. Construire l'indice batch de chaque point
        # ==============================================================
        #
        # Pour:
        #
        # B = 2
        #
        # on veut savoir quels points appartiennent:
        #
        # batch 0
        # batch 1
        #
        # afin de ne JAMAIS fusionner deux samples différents.
        #

        batch_indices = (
            torch.arange(
                B,
                device=features.device,
                dtype=torch.long,
            )
            .view(B, 1, 1, 1, 1)
            .expand(
                B,
                N,
                D,
                Hf,
                Wf,
            )
            .reshape(-1)
        )

        # ==============================================================
        # 3. XYZ continu -> voxel indices
        # ==============================================================
        #
        # Exemple X:
        #
        # ix =
        # floor(
        #     (X - Xmin) / dx
        # )
        #
        #
        # geometry_flat:
        #
        # [P,3]
        #
        # grid_min:
        #
        # [3]
        #
        # voxel_size:
        #
        # [3]
        #
        # broadcasting automatique.
        #

        voxel_indices = torch.floor(
            (
                geometry_flat
                - self.grid_min.to(
                    geometry_flat.dtype
                )
            )
            / self.voxel_size.to(
                geometry_flat.dtype
            )
        ).long()

        # voxel_indices:
        #
        # [P,3]
        #
        # avec:
        #
        # [:,0] = ix
        # [:,1] = iy
        # [:,2] = iz

        ix = voxel_indices[:, 0]
        iy = voxel_indices[:, 1]
        iz = voxel_indices[:, 2]

        # ==============================================================
        # 4. Supprimer les points hors de la grille
        # ==============================================================

        valid = (
            (ix >= 0)
            & (ix < Nx)
            & (iy >= 0)
            & (iy < Ny)
            & (iz >= 0)
            & (iz < Nz)
        )

        features_flat = features_flat[valid]

        ix = ix[valid]
        iy = iy[valid]
        iz = iz[valid]

        batch_indices = batch_indices[valid]

        # ==============================================================
        # 5. Construire un index unique par voxel
        # ==============================================================
        #
        # Il faut distinguer:
        #
        # batch
        # z
        # x
        # y
        #
        # On transforme donc:
        #
        # (b, iz, ix, iy)
        #
        # en un seul indice entier.
        #
        #
        # Layout choisi:
        #
        # [B, Nz, Nx, Ny]
        #
        # donc:
        #
        # linear_index =
        #
        # (((b * Nz + iz) * Nx + ix) * Ny + iy)
        #

        linear_indices = (
            (
                (
                    batch_indices * Nz
                    + iz
                )
                * Nx
                + ix
            )
            * Ny
            + iy
        )

        # ==============================================================
        # 6. Buffer de sortie aplati
        # ==============================================================
        #
        # Chaque voxel contient C features.
        #
        # Nombre total de voxels:
        #
        # B * Nz * Nx * Ny
        #

        num_voxels_total = (
            B
            * Nz
            * Nx
            * Ny
        )

        bev_flat = torch.zeros(
            num_voxels_total,
            C,
            dtype=features.dtype,
            device=features.device,
        )

        # ==============================================================
        # 7. SPLAT = somme des features dans chaque voxel
        # ==============================================================
        #
        # Si:
        #
        # point A -> voxel 52
        # point B -> voxel 52
        # point C -> voxel 52
        #
        # alors:
        #
        # bev_flat[52]
        #
        # =
        #
        # feature_A
        # + feature_B
        # + feature_C
        #
        #
        # index_add_ fait exactement cette opération.
        #

        if linear_indices.numel() > 0:
            bev_flat.index_add_(
                dim=0,
                index=linear_indices,
                source=features_flat,
            )

        # ==============================================================
        # 8. Restaurer les dimensions spatiales
        # ==============================================================
        #
        # [B*Nz*Nx*Ny, C]
        #
        # ↓
        #
        # [B,Nz,Nx,Ny,C]
        #

        bev = bev_flat.reshape(
            B,
            Nz,
            Nx,
            Ny,
            C,
        )

        # Puis mettre C juste après B:
        #
        # [B,C,Nz,Nx,Ny]
        #

        bev = bev.permute(
            0,  # B
            4,  # C
            1,  # Nz
            2,  # Nx
            3,  # Ny
        ).contiguous()

        # ==============================================================
        # 9. Collapse de Z
        # ==============================================================
        #
        # Cas actuel:
        #
        # Nz = 1
        #
        # [B,C,1,Nx,Ny]
        #
        # ↓
        #
        # [B,C,Nx,Ny]
        #

        if Nz == 1:
            bev = bev[:, :, 0]

        elif self.collapse_z:

            # Si Nz > 1:
            #
            # [B,C,Nz,Nx,Ny]
            #
            # ↓
            #
            # [B,C*Nz,Nx,Ny]
            #

            bev = bev.reshape(
                B,
                C * Nz,
                Nx,
                Ny,
            )

        # Si:
        #
        # Nz > 1
        # collapse_z = False
        #
        # on garde:
        #
        # [B,C,Nz,Nx,Ny]

        return bev