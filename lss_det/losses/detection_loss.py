#!/usr/bin/env python3

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Gather utility
# ======================================================================

def gather_feature_map(
    prediction: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    Extrait les prédictions uniquement aux positions des objets GT.

    Parameters
    ----------
    prediction:
        [B, C, Nx, Ny]

    indices:
        [B, M]

        index linéaire:

            index = ix * Ny + iy


    Returns
    -------
    gathered:
        [B, M, C]


    Exemple
    -------
    reg:

        [B,2,200,200]

            ↓ flatten spatial

        [B,40000,2]

            ↓ gather indices

        [B,M,2]
    """

    if prediction.ndim != 4:
        raise ValueError(
            "prediction must have shape [B,C,Nx,Ny]. "
            f"Received {tuple(prediction.shape)}"
        )

    if indices.ndim != 2:
        raise ValueError(
            "indices must have shape [B,M]. "
            f"Received {tuple(indices.shape)}"
        )

    B, C, Nx, Ny = prediction.shape

    if indices.shape[0] != B:
        raise ValueError(
            "prediction and indices batch sizes differ."
        )

    # --------------------------------------------------------------
    # [B,C,Nx,Ny]
    #
    #       ↓
    #
    # [B,Nx,Ny,C]
    #
    prediction = prediction.permute(
        0,
        2,
        3,
        1,
    ).contiguous()

    # --------------------------------------------------------------
    # [B,Nx,Ny,C]
    #
    #       ↓
    #
    # [B,Nx*Ny,C]
    #
    prediction = prediction.reshape(
        B,
        Nx * Ny,
        C,
    )

    # --------------------------------------------------------------
    # indices:
    #
    # [B,M]
    #
    #       ↓
    #
    # [B,M,C]
    #
    gather_indices = indices.unsqueeze(
        -1
    ).expand(
        -1,
        -1,
        C,
    )

    gathered = torch.gather(
        prediction,
        dim=1,
        index=gather_indices,
    )

    return gathered


# ======================================================================
# Gaussian focal loss
# ======================================================================

def gaussian_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    CenterNet-style Gaussian focal loss.

    logits:
        [B,K,Nx,Ny]

    target:
        [B,K,Nx,Ny]

    Target values:
        1.0     -> vrai centre objet
        0..1    -> Gaussian autour du centre
        0.0     -> background


    Important:
        On reçoit des LOGITS.

        Le sigmoid est appliqué ici.
    """

    if logits.shape != target.shape:
        raise ValueError(
            "Heatmap prediction and target must "
            "have identical shapes."
        )

    pred = torch.sigmoid(
        logits
    )

    # Évite:
    #
    # log(0)
    #
    pred = pred.clamp(
        min=eps,
        max=1.0 - eps,
    )

    # --------------------------------------------------------------
    # Positive positions
    #
    # Seulement les vrais centres ont target == 1.
    # --------------------------------------------------------------

    positive = (
        target == 1.0
    ).to(logits.dtype)

    # --------------------------------------------------------------
    # Negative / Gaussian surroundings
    # --------------------------------------------------------------

    negative = (
        target < 1.0
    ).to(logits.dtype)

    # Une cellule proche d'un centre possède par exemple:
    #
    # target = 0.8
    #
    # et doit être moins pénalisée qu'un vrai background:
    #
    # target = 0
    #
    negative_weight = torch.pow(
        1.0 - target,
        beta,
    )

    # --------------------------------------------------------------
    # Positive loss
    # --------------------------------------------------------------

    positive_loss = (
        torch.log(pred)
        * torch.pow(
            1.0 - pred,
            alpha,
        )
        * positive
    )

    # --------------------------------------------------------------
    # Negative loss
    # --------------------------------------------------------------

    negative_loss = (
        torch.log(
            1.0 - pred
        )
        * torch.pow(
            pred,
            alpha,
        )
        * negative_weight
        * negative
    )

    num_positive = positive.sum()

    positive_loss = positive_loss.sum()

    negative_loss = negative_loss.sum()

    # --------------------------------------------------------------
    # Cas sans objet dans l'image
    # --------------------------------------------------------------

    if num_positive < 1:

        return -negative_loss

    # Normalisation par nombre d'objets / centres.
    return -(
        positive_loss
        + negative_loss
    ) / num_positive


# ======================================================================
# Masked regression loss
# ======================================================================

def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    L1 loss uniquement sur les objets GT valides.

    prediction:
        [B,M,C]

    target:
        [B,M,C]

    mask:
        [B,M]

    Les slots padding de max_objects sont ignorés.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            "Regression prediction and target "
            "must have identical shapes."
        )

    if mask.shape != prediction.shape[:2]:
        raise ValueError(
            "Mask must have shape [B,M]."
        )

    mask_expanded = mask.unsqueeze(
        -1
    ).expand_as(
        prediction
    )

    mask_expanded = mask_expanded.to(
        prediction.dtype
    )

    absolute_error = torch.abs(
        prediction - target
    )

    absolute_error = (
        absolute_error
        * mask_expanded
    )

    # Nombre de valeurs scalaires réellement supervisées.
    denominator = mask_expanded.sum().clamp(
        min=1.0
    )

    return (
        absolute_error.sum()
        / denominator
    )


# ======================================================================
# Complete detection loss
# ======================================================================

class DetectionLoss(nn.Module):
    """
    Loss complète pour notre Center-based 3D detector.

    Total:

        heatmap loss
        +
        offset loss
        +
        center_z loss
        +
        dimension loss
        +
        rotation loss
    """

    def __init__(
        self,
        heatmap_weight: float = 1.0,
        reg_weight: float = 1.0,
        center_z_weight: float = 1.0,
        dim_weight: float = 1.0,
        rot_weight: float = 1.0,
    ):
        super().__init__()

        self.heatmap_weight = float(
            heatmap_weight
        )

        self.reg_weight = float(
            reg_weight
        )

        self.center_z_weight = float(
            center_z_weight
        )

        self.dim_weight = float(
            dim_weight
        )

        self.rot_weight = float(
            rot_weight
        )

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        # ==========================================================
        # 1. Heatmap
        # ==========================================================

        heatmap_loss = gaussian_focal_loss(
            logits=predictions["heatmap"],
            target=targets["heatmap"],
        )

        # ==========================================================
        # 2. Extract predictions at GT object centers
        # ==========================================================

        indices = targets[
            "indices"
        ]

        mask = targets[
            "mask"
        ]

        # ----------------------------------------------------------
        # reg
        # ----------------------------------------------------------

        reg_pred = gather_feature_map(
            predictions["reg"],
            indices,
        )

        # [B,M,2]

        # ----------------------------------------------------------
        # center_z
        # ----------------------------------------------------------

        center_z_pred = gather_feature_map(
            predictions["center_z"],
            indices,
        )

        # [B,M,1]

        # ----------------------------------------------------------
        # dim
        # ----------------------------------------------------------

        dim_pred = gather_feature_map(
            predictions["dim"],
            indices,
        )

        # [B,M,3]

        # ----------------------------------------------------------
        # rot
        # ----------------------------------------------------------

        rot_pred = gather_feature_map(
            predictions["rot"],
            indices,
        )

        # [B,M,2]

        # ==========================================================
        # 3. Regression losses
        # ==========================================================

        reg_loss = masked_l1_loss(
            prediction=reg_pred,
            target=targets["offset"],
            mask=mask,
        )

        center_z_loss = masked_l1_loss(
            prediction=center_z_pred,
            target=targets["center_z"],
            mask=mask,
        )

        dim_loss = masked_l1_loss(
            prediction=dim_pred,
            target=targets["dim"],
            mask=mask,
        )

        rot_loss = masked_l1_loss(
            prediction=rot_pred,
            target=targets["rot"],
            mask=mask,
        )

        # ==========================================================
        # 4. Weighted total
        # ==========================================================

        total_loss = (
            self.heatmap_weight
            * heatmap_loss

            +

            self.reg_weight
            * reg_loss

            +

            self.center_z_weight
            * center_z_loss

            +

            self.dim_weight
            * dim_loss

            +

            self.rot_weight
            * rot_loss
        )

        # ==========================================================
        # 5. Return detailed losses
        # ==========================================================

        return {
            "loss": total_loss,

            "loss_heatmap": heatmap_loss,

            "loss_reg": reg_loss,

            "loss_center_z": center_z_loss,

            "loss_dim": dim_loss,

            "loss_rot": rot_loss,
        }