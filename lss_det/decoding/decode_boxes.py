#!/usr/bin/env python3

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Feature gather
# ======================================================================

def gather_feature_map(
    feature_map: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    Récupère les valeurs d'une feature map aux positions demandées.

    Parameters
    ----------
    feature_map:
        [B,C,Nx,Ny]

    indices:
        [B,M]

        index linéaire:
            index = ix * Ny + iy

    Returns
    -------
    gathered:
        [B,M,C]
    """

    if feature_map.ndim != 4:
        raise ValueError(
            "feature_map must have shape [B,C,Nx,Ny]."
        )

    if indices.ndim != 2:
        raise ValueError(
            "indices must have shape [B,M]."
        )

    B, C, Nx, Ny = feature_map.shape

    # --------------------------------------------------------------
    # [B,C,Nx,Ny]
    #
    # ↓
    #
    # [B,Nx,Ny,C]
    # --------------------------------------------------------------

    x = feature_map.permute(
        0,
        2,
        3,
        1,
    ).contiguous()

    # --------------------------------------------------------------
    # [B,Nx,Ny,C]
    #
    # ↓
    #
    # [B,Nx*Ny,C]
    # --------------------------------------------------------------

    x = x.reshape(
        B,
        Nx * Ny,
        C,
    )

    # --------------------------------------------------------------
    # indices:
    #
    # [B,M]
    #
    # ↓
    #
    # [B,M,C]
    # --------------------------------------------------------------

    gather_indices = indices.unsqueeze(
        -1
    ).expand(
        -1,
        -1,
        C,
    )

    x = torch.gather(
        x,
        dim=1,
        index=gather_indices,
    )

    return x


# ======================================================================
# Local maximum suppression on heatmap
# ======================================================================

def local_maximum_suppression(
    heatmap: torch.Tensor,
    kernel_size: int = 3,
) -> torch.Tensor:
    """
    Garde uniquement les maxima locaux de la heatmap.

    Input:
        [B,K,Nx,Ny]

    Output:
        [B,K,Nx,Ny]

    Exemple:

        0.10  0.30  0.20
        0.40  0.95  0.60
        0.20  0.50  0.30

    devient:

        0     0     0
        0    0.95   0
        0     0     0
    """

    if kernel_size % 2 == 0:
        raise ValueError(
            "kernel_size must be odd."
        )

    padding = (
        kernel_size - 1
    ) // 2

    local_max = F.max_pool2d(
        heatmap,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )

    keep = (
        heatmap == local_max
    ).to(
        heatmap.dtype
    )

    return heatmap * keep


# ======================================================================
# Circle NMS
# ======================================================================

def circle_nms(
    centers_xy: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    min_distance: float,
) -> torch.Tensor:
    """
    NMS simple basée sur la distance entre centres BEV.

    Elle est effectuée classe par classe.

    Parameters
    ----------
    centers_xy:
        [M,2] métrique, en mètres.

    scores:
        [M]

    labels:
        [M]

    min_distance:
        Distance minimale entre deux centres de même classe.

    Returns
    -------
    keep:
        indices des boxes conservées.
    """

    if centers_xy.shape[0] == 0:
        return torch.empty(
            0,
            dtype=torch.long,
            device=centers_xy.device,
        )

    keep_indices = []

    unique_labels = torch.unique(
        labels
    )

    for class_id in unique_labels:

        class_mask = (
            labels == class_id
        )

        class_indices = torch.nonzero(
            class_mask,
            as_tuple=False,
        ).squeeze(1)

        class_scores = scores[
            class_indices
        ]

        # Plus fort score d'abord.

        order = torch.argsort(
            class_scores,
            descending=True,
        )

        class_indices = class_indices[
            order
        ]

        while class_indices.numel() > 0:

            current = class_indices[0]

            keep_indices.append(
                current
            )

            if class_indices.numel() == 1:
                break

            remaining = class_indices[
                1:
            ]

            current_center = centers_xy[
                current
            ]

            remaining_centers = centers_xy[
                remaining
            ]

            distance = torch.linalg.norm(
                remaining_centers
                - current_center,
                dim=1,
            )

            # Garder uniquement les centres
            # suffisamment éloignés.

            class_indices = remaining[
                distance >= min_distance
            ]

    if len(keep_indices) == 0:
        return torch.empty(
            0,
            dtype=torch.long,
            device=centers_xy.device,
        )

    keep = torch.stack(
        keep_indices
    )

    # Remettre toutes les classes par score global.

    order = torch.argsort(
        scores[keep],
        descending=True,
    )

    return keep[order]


# ======================================================================
# Decoder
# ======================================================================

class CenterPointDecoder(nn.Module):
    """
    Decode les sorties denses de DetectionHead en boxes 3D.

    ------------------------------------------------------------------
    INPUT
    ------------------------------------------------------------------

    predictions["heatmap"]:
        [B,K,Nx,Ny]

    predictions["reg"]:
        [B,2,Nx,Ny]

    predictions["center_z"]:
        [B,1,Nx,Ny]

    predictions["dim"]:
        [B,3,Nx,Ny]

    predictions["rot"]:
        [B,2,Nx,Ny]


    ------------------------------------------------------------------
    OUTPUT
    ------------------------------------------------------------------

    Une liste de B dictionnaires:

        {
            "boxes":  [M,7],
            "scores": [M],
            "labels": [M],
        }

    boxes:

        [x,y,z,l,w,h,yaw]
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
        top_k: int = 100,
        score_threshold: float = 0.1,
        local_max_kernel: int = 3,
        use_circle_nms: bool = True,
        nms_min_distance: float = 1.0,
    ):
        super().__init__()

        self.x_min = float(
            xbound[0]
        )

        self.dx = float(
            xbound[2]
        )

        self.y_min = float(
            ybound[0]
        )

        self.dy = float(
            ybound[2]
        )

        self.top_k = int(
            top_k
        )

        self.score_threshold = float(
            score_threshold
        )

        self.local_max_kernel = int(
            local_max_kernel
        )

        self.use_circle_nms = bool(
            use_circle_nms
        )

        self.nms_min_distance = float(
            nms_min_distance
        )

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:

        # ==========================================================
        # 1. Check
        # ==========================================================

        required_keys = {
            "heatmap",
            "reg",
            "center_z",
            "dim",
            "rot",
        }

        missing = (
            required_keys
            - set(predictions.keys())
        )

        if missing:
            raise ValueError(
                f"Missing prediction keys: {missing}"
            )

        heatmap_logits = predictions[
            "heatmap"
        ]

        B, K, Nx, Ny = heatmap_logits.shape

        # ==========================================================
        # 2. Logits -> probabilities
        # ==========================================================

        heatmap = torch.sigmoid(
            heatmap_logits
        )

        # ==========================================================
        # 3. Garder uniquement maxima locaux
        # ==========================================================

        heatmap = local_maximum_suppression(
            heatmap,
            kernel_size=self.local_max_kernel,
        )

        # ==========================================================
        # 4. Flatten classes + spatial
        # ==========================================================
        #
        # [B,K,Nx,Ny]
        #
        # ↓
        #
        # [B,K*Nx*Ny]
        #

        heatmap_flat = heatmap.reshape(
            B,
            -1,
        )

        num_candidates = (
            K * Nx * Ny
        )

        top_k = min(
            self.top_k,
            num_candidates,
        )

        # ==========================================================
        # 5. Top-K global
        # ==========================================================

        scores, top_indices = torch.topk(
            heatmap_flat,
            k=top_k,
            dim=1,
        )

        # top_indices contient:
        #
        # class + spatial index
        #
        # puisque flatten:
        #
        # [K,Nx,Ny]

        spatial_size = (
            Nx * Ny
        )

        labels = (
            top_indices
            // spatial_size
        )

        spatial_indices = (
            top_indices
            % spatial_size
        )

        # ==========================================================
        # 6. Spatial index -> ix, iy
        # ==========================================================

        ix = (
            spatial_indices
            // Ny
        )

        iy = (
            spatial_indices
            % Ny
        )

        # ix / iy:
        #
        # [B,top_k]

        # ==========================================================
        # 7. Gather regression predictions
        # ==========================================================

        reg = gather_feature_map(
            predictions["reg"],
            spatial_indices,
        )

        center_z = gather_feature_map(
            predictions["center_z"],
            spatial_indices,
        )

        dims = gather_feature_map(
            predictions["dim"],
            spatial_indices,
        )

        rot = gather_feature_map(
            predictions["rot"],
            spatial_indices,
        )

        # Shapes:
        #
        # reg:
        # [B,Ktop,2]
        #
        # z:
        # [B,Ktop,1]
        #
        # dim:
        # [B,Ktop,3]
        #
        # rot:
        # [B,Ktop,2]

        # ==========================================================
        # 8. Cell + offset -> grid coordinates
        # ==========================================================

        gx = (
            ix.to(reg.dtype)
            + reg[..., 0]
        )

        gy = (
            iy.to(reg.dtype)
            + reg[..., 1]
        )

        # ==========================================================
        # 9. Grid -> metric coordinates
        # ==========================================================

        x = (
            self.x_min
            + gx * self.dx
        )

        y = (
            self.y_min
            + gy * self.dy
        )

        z = center_z[
            ...,
            0,
        ]

        # ==========================================================
        # 10. Dimensions
        # ==========================================================

        length = dims[
            ...,
            0,
        ]

        width = dims[
            ...,
            1,
        ]

        height = dims[
            ...,
            2,
        ]

        # ==========================================================
        # 11. Rotation
        # ==========================================================

        sin_yaw = rot[
            ...,
            0,
        ]

        cos_yaw = rot[
            ...,
            1,
        ]

        yaw = torch.atan2(
            sin_yaw,
            cos_yaw,
        )

        # ==========================================================
        # 12. Construire boxes
        # ==========================================================

        boxes = torch.stack(
            (
                x,
                y,
                z,
                length,
                width,
                height,
                yaw,
            ),
            dim=-1,
        )

        # boxes:
        #
        # [B,top_k,7]

        # ==========================================================
        # 13. Batch-by-batch filtering
        # ==========================================================

        decoded = []

        for b in range(B):

            boxes_b = boxes[b]

            scores_b = scores[b]

            labels_b = labels[b]

            # ------------------------------------------------------
            # Score threshold
            # ------------------------------------------------------

            valid = (
                scores_b
                >= self.score_threshold
            )

            # ------------------------------------------------------
            # On rejette également les dimensions non physiques.
            # ------------------------------------------------------

            valid_dims = (
                boxes_b[:, 3] > 0.0
            ) & (
                boxes_b[:, 4] > 0.0
            ) & (
                boxes_b[:, 5] > 0.0
            )

            valid = (
                valid
                & valid_dims
            )

            boxes_b = boxes_b[
                valid
            ]

            scores_b = scores_b[
                valid
            ]

            labels_b = labels_b[
                valid
            ]

            # ------------------------------------------------------
            # Circle NMS
            # ------------------------------------------------------

            if (
                self.use_circle_nms
                and
                boxes_b.shape[0] > 0
            ):

                keep = circle_nms(
                    centers_xy=boxes_b[
                        :,
                        0:2,
                    ],
                    scores=scores_b,
                    labels=labels_b,
                    min_distance=(
                        self.nms_min_distance
                    ),
                )

                boxes_b = boxes_b[
                    keep
                ]

                scores_b = scores_b[
                    keep
                ]

                labels_b = labels_b[
                    keep
                ]

            decoded.append(
                {
                    "boxes": boxes_b,
                    "scores": scores_b,
                    "labels": labels_b,
                }
            )

        return decoded