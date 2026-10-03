#!/usr/bin/env python3

from typing import Dict

import torch
import torch.nn as nn


# ==============================================================
# GroupNorm utility
# ==============================================================

def make_group_norm(
    num_channels: int,
) -> nn.GroupNorm:

    for num_groups in [
        32,
        16,
        8,
        4,
        2,
        1,
    ]:

        if (
            num_channels
            % num_groups
            == 0
        ):

            return nn.GroupNorm(
                num_groups=num_groups,
                num_channels=num_channels,
            )

    raise RuntimeError(
        f"Unable to create GroupNorm "
        f"for {num_channels} channels."
    )


# ==============================================================
# Prediction Head
# ==============================================================

class PredictionHead(nn.Module):

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

            make_group_norm(
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

        # ======================================================
        # Final conv initialization
        # ======================================================

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
        x,
    ):

        return self.net(
            x
        )


# ==============================================================
# Detection Head
# ==============================================================

class DetectionHead(nn.Module):

    def __init__(
        self,
        in_channels: int = 128,
        shared_channels: int = 128,
        head_channels: int = 64,
        num_classes: int = 6,
    ):
        super().__init__()

        self.in_channels = int(
            in_channels
        )

        self.num_classes = int(
            num_classes
        )

        # ======================================================
        # Shared BEV representation
        # ======================================================

        self.shared = nn.Sequential(

            nn.Conv2d(

                in_channels=in_channels,

                out_channels=shared_channels,

                kernel_size=3,

                padding=1,

                bias=False,
            ),

            make_group_norm(
                shared_channels
            ),

            nn.ReLU(
                inplace=True
            ),
        )

        # ======================================================
        # Heatmap
        # ======================================================

        self.heatmap_head = PredictionHead(

            in_channels=shared_channels,

            hidden_channels=head_channels,

            out_channels=num_classes,

            final_bias=-2.19,
        )

        # ======================================================
        # XY offset
        # ======================================================

        self.reg_head = PredictionHead(

            in_channels=shared_channels,

            hidden_channels=head_channels,

            out_channels=2,
        )

        # ======================================================
        # Z
        # ======================================================

        self.center_z_head = PredictionHead(

            in_channels=shared_channels,

            hidden_channels=head_channels,

            out_channels=1,
        )

        # ======================================================
        # L/W/H
        # ======================================================

        self.dim_head = PredictionHead(

            in_channels=shared_channels,

            hidden_channels=head_channels,

            out_channels=3,
        )

        # ======================================================
        # yaw
        # ======================================================

        self.rot_head = PredictionHead(

            in_channels=shared_channels,

            hidden_channels=head_channels,

            out_channels=2,
        )

    def forward(
        self,
        bev_features,
    ) -> Dict[
        str,
        torch.Tensor,
    ]:

        if (
            bev_features.ndim
            != 4
        ):

            raise ValueError(
                "DetectionHead expects "
                "[B,C,Nx,Ny]."
            )

        if (
            bev_features.shape[1]
            != self.in_channels
        ):

            raise ValueError(
                f"Expected "
                f"{self.in_channels} channels, "
                f"received "
                f"{bev_features.shape[1]}."
            )

        # ======================================================
        # Shared features
        # ======================================================

        x = self.shared(
            bev_features
        )

        # ======================================================
        # Heads
        # ======================================================

        return {

            "heatmap":
                self.heatmap_head(
                    x
                ),

            "reg":
                self.reg_head(
                    x
                ),

            "center_z":
                self.center_z_head(
                    x
                ),

            "dim":
                self.dim_head(
                    x
                ),

            "rot":
                self.rot_head(
                    x
                ),
        }