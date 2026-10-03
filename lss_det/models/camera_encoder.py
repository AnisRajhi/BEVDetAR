#!/usr/bin/env python3

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet


# ==============================================================
# GroupNorm utility
# ==============================================================

def make_group_norm(
    num_channels: int,
) -> nn.GroupNorm:
    """
    Choisit automatiquement un nombre de groupes valide.

    On privilégie 32 groupes, puis 16, 8, ...
    """

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
        f"Could not create GroupNorm "
        f"for {num_channels} channels."
    )


# ==============================================================
# Feature Fusion
# ==============================================================

class FeatureFusion(nn.Module):
    """
    Fusion de:

        EfficientNet reduction_4
            [B*N,112,8,22]

    et:

        EfficientNet reduction_5
            [B*N,320,4,11]

    Sortie:

        [B*N,512,8,22]


    IMPORTANT
    ---------

    On utilise GroupNorm et non BatchNorm.

    GroupNorm ne dépend pas du batch size.
    """

    def __init__(
        self,
        out_channels: int = 512,
    ):
        super().__init__()

        self.out_channels = (
            out_channels
        )

        self.fusion = nn.Sequential(

            # --------------------------------------------------
            # 112 + 320 = 432 channels
            # --------------------------------------------------

            nn.Conv2d(
                in_channels=(
                    112 + 320
                ),
                out_channels=512,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            make_group_norm(
                512
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                in_channels=512,
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

    def forward(
        self,
        reduction_4: torch.Tensor,
        reduction_5: torch.Tensor,
    ) -> torch.Tensor:

        # ======================================================
        # Upsample deep EfficientNet features
        # ======================================================

        reduction_5 = F.interpolate(

            reduction_5,

            size=reduction_4.shape[
                -2:
            ],

            mode="bilinear",

            align_corners=True,
        )

        # ======================================================
        # Concatenate
        # ======================================================

        x = torch.cat(
            [
                reduction_4,
                reduction_5,
            ],
            dim=1,
        )

        # [BN,432,8,22]

        x = self.fusion(
            x
        )

        # [BN,512,8,22]

        return x


# ==============================================================
# Camera Encoder
# ==============================================================

class CameraEncoder(nn.Module):
    """
    Shared camera encoder.

    INPUT
    -----

        images:
            [B,N,3,H,W]

    OUTPUT
    ------

        [B,N,512,H/16,W/16]


    NORMALIZATION POLICY
    --------------------

    EfficientNet pretrained BatchNorm:
        frozen in eval mode.

    Our fusion layers:
        GroupNorm.

    EfficientNet DropConnect:
        disabled.

    Cela rend le CameraEncoder beaucoup plus robuste
    lorsque le training utilise batch_size=1.
    """

    def __init__(
        self,
        out_channels: int = 512,
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        freeze_backbone_bn: bool = True,
        disable_drop_connect: bool = True,
    ):
        super().__init__()

        self.out_channels = int(
            out_channels
        )

        self.freeze_backbone_bn = bool(
            freeze_backbone_bn
        )

        # ======================================================
        # EfficientNet-B0
        # ======================================================

        if pretrained:

            if weights_path is None:

                self.backbone = (
                    EfficientNet.from_pretrained(
                        "efficientnet-b0"
                    )
                )

            else:

                self.backbone = (
                    EfficientNet.from_pretrained(
                        "efficientnet-b0",
                        weights_path=weights_path,
                    )
                )

        else:

            self.backbone = (
                EfficientNet.from_name(
                    "efficientnet-b0"
                )
            )

        # ======================================================
        # Disable DropConnect / stochastic depth
        # ======================================================
        #
        # EfficientNet utilise du drop-connect à l'intérieur
        # des MBConv blocks en mode train().
        #
        # Pour notre petit batch et pour obtenir un comportement
        # train/eval comparable, on le désactive.
        #
        # ======================================================

        if disable_drop_connect:

            global_params = (
                self.backbone
                ._global_params
            )

            if hasattr(
                global_params,
                "_replace",
            ):

                self.backbone._global_params = (
                    global_params._replace(
                        drop_connect_rate=0.0
                    )
                )

        # ======================================================
        # Fusion
        # ======================================================

        self.fusion = FeatureFusion(
            out_channels=out_channels
        )

        # ======================================================
        # Freeze EfficientNet BN statistics
        # ======================================================

        if self.freeze_backbone_bn:

            self._freeze_backbone_batchnorm()

    # ==========================================================
    # Freeze EfficientNet BatchNorm
    # ==========================================================

    def _freeze_backbone_batchnorm(
        self,
    ):
        """
        BatchNorm EfficientNet:

            - reste en eval()
            - running_mean / running_var ne changent plus
            - gamma/beta sont également figés

        Les convolutions EfficientNet restent entraînables.
        """

        for module in (
            self.backbone.modules()
        ):

            if isinstance(
                module,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                ),
            ):

                module.eval()

                if (
                    module.weight
                    is not None
                ):

                    module.weight.requires_grad_(
                        False
                    )

                if (
                    module.bias
                    is not None
                ):

                    module.bias.requires_grad_(
                        False
                    )

    # ==========================================================
    # train()
    # ==========================================================

    def train(
        self,
        mode: bool = True,
    ):
        """
        Le problème:

            model.train()

        remet normalement TOUTES les BatchNorm en train mode.

        On surcharge donc train() pour laisser les BN
        EfficientNet en eval mode.
        """

        super().train(
            mode
        )

        if self.freeze_backbone_bn:

            self._freeze_backbone_batchnorm()

        return self

    # ==========================================================
    # Forward
    # ==========================================================

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:

        if images.ndim != 5:

            raise ValueError(
                "CameraEncoder expects "
                "[B,N,3,H,W]. "
                f"Received "
                f"{tuple(images.shape)}"
            )

        B, N, C, H, W = (
            images.shape
        )

        if C != 3:

            raise ValueError(
                "Expected RGB images."
            )

        # ======================================================
        # Merge B and N
        # ======================================================

        x = images.reshape(
            B * N,
            C,
            H,
            W,
        )

        # ======================================================
        # EfficientNet
        # ======================================================

        endpoints = (
            self.backbone
            .extract_endpoints(
                x
            )
        )

        reduction_4 = endpoints[
            "reduction_4"
        ]

        reduction_5 = endpoints[
            "reduction_5"
        ]

        # ======================================================
        # Feature fusion
        # ======================================================

        features = self.fusion(

            reduction_4,

            reduction_5,
        )

        _, Cout, Hf, Wf = (
            features.shape
        )

        # ======================================================
        # Restore B/N
        # ======================================================

        features = features.reshape(

            B,

            N,

            Cout,

            Hf,

            Wf,
        )

        return features