#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


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
# Residual Block
# ==============================================================

class BasicBlock(nn.Module):
    """
    ResNet-like residual block.

    BatchNorm a été remplacé par GroupNorm.

    Cela signifie que le comportement ne dépend plus
    des statistiques du batch.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()

        # ======================================================
        # Main branch
        # ======================================================

        self.conv1 = nn.Conv2d(

            in_channels=in_channels,

            out_channels=out_channels,

            kernel_size=3,

            stride=stride,

            padding=1,

            bias=False,
        )

        self.norm1 = make_group_norm(
            out_channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.conv2 = nn.Conv2d(

            in_channels=out_channels,

            out_channels=out_channels,

            kernel_size=3,

            stride=1,

            padding=1,

            bias=False,
        )

        self.norm2 = make_group_norm(
            out_channels
        )

        # ======================================================
        # Residual branch
        # ======================================================

        if (
            stride != 1
            or
            in_channels
            != out_channels
        ):

            self.downsample = nn.Sequential(

                nn.Conv2d(

                    in_channels=in_channels,

                    out_channels=out_channels,

                    kernel_size=1,

                    stride=stride,

                    bias=False,
                ),

                make_group_norm(
                    out_channels
                ),
            )

        else:

            self.downsample = None

    def forward(
        self,
        x,
    ):

        identity = x

        out = self.conv1(
            x
        )

        out = self.norm1(
            out
        )

        out = self.relu(
            out
        )

        out = self.conv2(
            out
        )

        out = self.norm2(
            out
        )

        if (
            self.downsample
            is not None
        ):

            identity = (
                self.downsample(
                    identity
                )
            )

        out = (
            out
            + identity
        )

        out = self.relu(
            out
        )

        return out


# ==============================================================
# BEV Backbone
# ==============================================================

class BEVBackbone(nn.Module):

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 128,
    ):
        super().__init__()

        self.in_channels = int(
            in_channels
        )

        self.out_channels = int(
            out_channels
        )

        # ======================================================
        # Stem
        # ======================================================

        self.stem = nn.Sequential(

            nn.Conv2d(

                in_channels=in_channels,

                out_channels=64,

                kernel_size=7,

                stride=2,

                padding=3,

                bias=False,
            ),

            make_group_norm(
                64
            ),

            nn.ReLU(
                inplace=True
            ),
        )

        # ======================================================
        # ResNet stages
        # ======================================================

        self.layer1 = self._make_layer(

            in_channels=64,

            out_channels=64,

            num_blocks=2,

            stride=1,
        )

        self.layer2 = self._make_layer(

            in_channels=64,

            out_channels=128,

            num_blocks=2,

            stride=2,
        )

        self.layer3 = self._make_layer(

            in_channels=128,

            out_channels=256,

            num_blocks=2,

            stride=2,
        )

        # ======================================================
        # Deep/local feature fusion
        # ======================================================

        self.fusion = nn.Sequential(

            nn.Conv2d(

                in_channels=(
                    256 + 64
                ),

                out_channels=256,

                kernel_size=3,

                padding=1,

                bias=False,
            ),

            make_group_norm(
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(

                in_channels=256,

                out_channels=256,

                kernel_size=3,

                padding=1,

                bias=False,
            ),

            make_group_norm(
                256
            ),

            nn.ReLU(
                inplace=True
            ),
        )

        # ======================================================
        # Output projection
        # ======================================================

        self.output_block = nn.Sequential(

            nn.Conv2d(

                in_channels=256,

                out_channels=128,

                kernel_size=3,

                padding=1,

                bias=False,
            ),

            make_group_norm(
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(

                in_channels=128,

                out_channels=out_channels,

                kernel_size=3,

                padding=1,

                bias=False,
            ),

            make_group_norm(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),
        )

        self._initialize_weights()

    # ==========================================================
    # Make residual stage
    # ==========================================================

    @staticmethod
    def _make_layer(
        in_channels,
        out_channels,
        num_blocks,
        stride,
    ):

        blocks = [

            BasicBlock(

                in_channels=in_channels,

                out_channels=out_channels,

                stride=stride,
            )
        ]

        for _ in range(
            1,
            num_blocks,
        ):

            blocks.append(

                BasicBlock(

                    in_channels=out_channels,

                    out_channels=out_channels,

                    stride=1,
                )
            )

        return nn.Sequential(
            *blocks
        )

    # ==========================================================
    # Initialization
    # ==========================================================

    def _initialize_weights(
        self,
    ):

        for module in self.modules():

            if isinstance(
                module,
                nn.Conv2d,
            ):

                nn.init.kaiming_normal_(

                    module.weight,

                    mode="fan_out",

                    nonlinearity="relu",
                )

            elif isinstance(
                module,
                nn.GroupNorm,
            ):

                nn.init.constant_(

                    module.weight,

                    1.0,
                )

                nn.init.constant_(

                    module.bias,

                    0.0,
                )

        # ======================================================
        # Start residual branches close to identity
        # ======================================================

        for module in self.modules():

            if isinstance(
                module,
                BasicBlock,
            ):

                nn.init.constant_(

                    module.norm2.weight,

                    0.0,
                )

    # ==========================================================
    # Forward
    # ==========================================================

    def forward(
        self,
        bev,
    ):

        if bev.ndim != 4:

            raise ValueError(
                "Expected [B,C,Nx,Ny]."
            )

        if (
            bev.shape[1]
            != self.in_channels
        ):

            raise ValueError(
                f"Expected "
                f"{self.in_channels} channels, "
                f"received "
                f"{bev.shape[1]}."
            )

        input_size = (
            bev.shape[-2:]
        )

        # ======================================================
        # Encoder
        # ======================================================

        x = self.stem(
            bev
        )

        x1 = self.layer1(
            x
        )

        x2 = self.layer2(
            x1
        )

        x3 = self.layer3(
            x2
        )

        # ======================================================
        # Deep feature → x1 resolution
        # ======================================================

        x3_up = F.interpolate(

            x3,

            size=x1.shape[
                -2:
            ],

            mode="bilinear",

            align_corners=True,
        )

        # ======================================================
        # Skip fusion
        # ======================================================

        x = torch.cat(

            [
                x1,
                x3_up,
            ],

            dim=1,
        )

        x = self.fusion(
            x
        )

        # ======================================================
        # Back to original BEV resolution
        # ======================================================

        x = F.interpolate(

            x,

            size=input_size,

            mode="bilinear",

            align_corners=True,
        )

        x = self.output_block(
            x
        )

        return x