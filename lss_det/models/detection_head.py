#!/usr/bin/env python3

from typing import Dict

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    """
    Petite tête de prédiction utilisée pour une grandeur donnée.

    Exemple:
        input  : [B,128,H,W]
        output : [B,2,H,W]

    pour reg = (dx, dy).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        final_bias: float = 0.0,
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                hidden_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=True,
            ),
        )

        # Initialisation de la dernière couche.
        nn.init.normal_(
            self.net[-1].weight,
            mean=0.0,
            std=0.001,
        )

        nn.init.constant_(
            self.net[-1].bias,
            final_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.net(x)


class DetectionHead(nn.Module):
    """
    Center-based 3D Detection Head.

    ------------------------------------------------------------------
    INPUT
    ------------------------------------------------------------------

    bev_features:
        [B,128,Nx,Ny]

    Exemple:
        [B,128,200,200]


    ------------------------------------------------------------------
    OUTPUTS
    ------------------------------------------------------------------

    heatmap:
        [B,K,Nx,Ny]

        Logits de présence du centre d'un objet
        pour chaque classe.

    reg:
        [B,2,Nx,Ny]

        0 = dx
        1 = dy

    center_z:
        [B,1,Nx,Ny]

        Coordonnée Z du centre de la box.

    dim:
        [B,3,Nx,Ny]

        0 = length
        1 = width
        2 = height

    rot:
        [B,2,Nx,Ny]

        0 = sin(yaw)
        1 = cos(yaw)


    IMPORTANT
    ---------
    heatmap contient des LOGITS.

    Pas de sigmoid ici.

    Le sigmoid sera utilisé:
        - dans la loss
        - dans le decoder
        - pour la visualisation
    """

    def __init__(
        self,
        in_channels: int = 128,
        shared_channels: int = 128,
        head_channels: int = 64,
        num_classes: int = 6,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # ==========================================================
        # 1. Shared feature processing
        # ==========================================================
        #
        # Toutes les têtes reçoivent une représentation BEV commune.
        #
        # [B,128,H,W]
        #
        #       ↓
        #
        # [B,128,H,W]
        #

        self.shared = nn.Sequential(

            nn.Conv2d(
                in_channels=in_channels,
                out_channels=shared_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                shared_channels
            ),

            nn.ReLU(
                inplace=True
            ),
        )

        # ==========================================================
        # 2. Heatmap head
        # ==========================================================
        #
        # K channels:
        #
        # car
        # truck
        # pedestrian
        # ...
        #
        # Sortie:
        #
        # [B,K,H,W]
        #
        #
        # Bias négatif:
        #
        # au début du training, on veut une faible probabilité
        # d'objet partout.
        #
        # sigmoid(-2.19) ≈ 0.10
        #

        self.heatmap_head = PredictionHead(
            in_channels=shared_channels,
            hidden_channels=head_channels,
            out_channels=num_classes,
            final_bias=-2.19,
        )

        # ==========================================================
        # 3. Center offset head
        # ==========================================================

        self.reg_head = PredictionHead(
            in_channels=shared_channels,
            hidden_channels=head_channels,
            out_channels=2,
        )

        # ==========================================================
        # 4. Z center
        # ==========================================================

        self.center_z_head = PredictionHead(
            in_channels=shared_channels,
            hidden_channels=head_channels,
            out_channels=1,
        )

        # ==========================================================
        # 5. Box dimensions
        # ==========================================================

        self.dim_head = PredictionHead(
            in_channels=shared_channels,
            hidden_channels=head_channels,
            out_channels=3,
        )

        # ==========================================================
        # 6. Orientation
        # ==========================================================

        self.rot_head = PredictionHead(
            in_channels=shared_channels,
            hidden_channels=head_channels,
            out_channels=2,
        )

    def forward(
        self,
        bev_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        # ----------------------------------------------------------
        # Input validation
        # ----------------------------------------------------------

        if bev_features.ndim != 4:

            raise ValueError(
                "DetectionHead expects "
                "[B,C,Nx,Ny]. "
                f"Received {tuple(bev_features.shape)}"
            )

        if bev_features.shape[1] != self.in_channels:

            raise ValueError(
                f"DetectionHead expects "
                f"{self.in_channels} channels, "
                f"received {bev_features.shape[1]}."
            )

        # ==========================================================
        # Shared features
        # ==============================================================

        x = self.shared(
            bev_features
        )

        # x:
        #
        # [B,shared_channels,Nx,Ny]

        # ==========================================================
        # Independent prediction heads
        # ==============================================================

        heatmap = self.heatmap_head(x)

        reg = self.reg_head(x)

        center_z = self.center_z_head(x)

        dim = self.dim_head(x)

        rot = self.rot_head(x)

        # ==========================================================
        # Return raw predictions
        # ==============================================================

        return {
            "heatmap": heatmap,
            "reg": reg,
            "center_z": center_z,
            "dim": dim,
            "rot": rot,
        }


# ======================================================================
# Manual test
# ======================================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = DetectionHead(
        in_channels=128,
        shared_channels=128,
        head_channels=64,
        num_classes=6,
    ).to(device)

    bev_features = torch.randn(
        2,
        128,
        200,
        200,
        device=device,
    )

    with torch.no_grad():

        predictions = model(
            bev_features
        )

    print(
        "Heatmap :",
        predictions["heatmap"].shape,
    )

    print(
        "Reg     :",
        predictions["reg"].shape,
    )

    print(
        "Center Z:",
        predictions["center_z"].shape,
    )

    print(
        "Dim     :",
        predictions["dim"].shape,
    )

    print(
        "Rot     :",
        predictions["rot"].shape,
    )

    assert predictions["heatmap"].shape == (
        2,
        6,
        200,
        200,
    )

    assert predictions["reg"].shape == (
        2,
        2,
        200,
        200,
    )

    assert predictions["center_z"].shape == (
        2,
        1,
        200,
        200,
    )

    assert predictions["dim"].shape == (
        2,
        3,
        200,
        200,
    )

    assert predictions["rot"].shape == (
        2,
        2,
        200,
        200,
    )

    print(
        "DetectionHead test PASSED"
    )