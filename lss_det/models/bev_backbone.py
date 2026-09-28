#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# Basic ResNet Block
# ==============================================================

class BasicBlock(nn.Module):
    """
    Bloc résiduel de type ResNet-18.

    Exemple sans changement de résolution:

        [B,64,H,W]
             │
             ├──────────── identity ────────────┐
             │                                  │
             ↓                                  │
        Conv 3x3                                │
        BN                                      │
        ReLU                                    │
             ↓                                  │
        Conv 3x3                                │
        BN                                      │
             ↓                                  │
             +  <───────────────────────────────┘
             ↓
           ReLU


    Si stride=2 ou si le nombre de channels change,
    l'identity passe également par une Conv 1x1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()

        # ----------------------------------------------------------
        # Branche principale
        # ----------------------------------------------------------

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
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

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        # ----------------------------------------------------------
        # Branche résiduelle / shortcut
        # ----------------------------------------------------------
        #
        # Si:
        #
        # in_channels != out_channels
        #
        # ou:
        #
        # stride != 1
        #
        # alors x et la branche principale n'ont plus
        # la même shape.
        #
        # Il faut donc adapter l'identity.
        #

        if (
            stride != 1
            or in_channels != out_channels
        ):

            self.downsample = nn.Sequential(

                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),

                nn.BatchNorm2d(
                    out_channels
                ),
            )

        else:

            self.downsample = None

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        identity = x

        # Branche principale

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Adapter l'identity si nécessaire

        if self.downsample is not None:
            identity = self.downsample(
                identity
            )

        # Résidual connection

        out = out + identity

        out = self.relu(out)

        return out


# ==============================================================
# BEV Backbone
# ==============================================================

class BEVBackbone(nn.Module):
    """
    Backbone travaillant sur les features BEV produites par SPLAT.

    ----------------------------------------------------------------
    INPUT
    ----------------------------------------------------------------

    bev:
        [B, in_channels, Nx, Ny]

    Configuration actuelle:

        [B,64,200,200]


    ----------------------------------------------------------------
    OUTPUT
    ----------------------------------------------------------------

        [B,out_channels,Nx,Ny]

    Configuration actuelle:

        [B,128,200,200]


    ----------------------------------------------------------------
    ARCHITECTURE
    ----------------------------------------------------------------

    [B,64,200,200]

        ↓ stem stride 2

    [B,64,100,100]

        ↓ layer1

    x1:
    [B,64,100,100]

        ↓ layer2 stride 2

    [B,128,50,50]

        ↓ layer3 stride 2

    x3:
    [B,256,25,25]

        ↓ upsample vers x1

    [B,256,100,100]

        ↓ concat x1

    [B,320,100,100]

        ↓ fusion

    [B,256,100,100]

        ↓ upsample

    [B,256,200,200]

        ↓ output block

    [B,128,200,200]
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 128,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # ==========================================================
        # 1. Stem
        # ==========================================================
        #
        # Contrairement au ResNet-18 standard,
        # on ne met PAS de max-pooling ici.
        #
        # C'est cohérent avec le BevEncode de LSS.
        #

        self.stem = nn.Sequential(

            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),
        )

        # ==========================================================
        # 2. ResNet stages
        # ==========================================================

        # ----------------------------------------------------------
        # layer1
        #
        # [B,64,100,100]
        #
        # résolution inchangée
        # ----------------------------------------------------------

        self.layer1 = self._make_layer(
            in_channels=64,
            out_channels=64,
            num_blocks=2,
            stride=1,
        )

        # ----------------------------------------------------------
        # layer2
        #
        # [B,64,100,100]
        #
        #        ↓
        #
        # [B,128,50,50]
        # ----------------------------------------------------------

        self.layer2 = self._make_layer(
            in_channels=64,
            out_channels=128,
            num_blocks=2,
            stride=2,
        )

        # ----------------------------------------------------------
        # layer3
        #
        # [B,128,50,50]
        #
        #        ↓
        #
        # [B,256,25,25]
        # ----------------------------------------------------------

        self.layer3 = self._make_layer(
            in_channels=128,
            out_channels=256,
            num_blocks=2,
            stride=2,
        )

        # ==========================================================
        # 3. Fusion deep features + skip connection
        # ==========================================================
        #
        # Deep:
        #   256 channels
        #
        # Skip x1:
        #   64 channels
        #
        # Concat:
        #
        #   256 + 64 = 320
        #
        # Puis:
        #
        #   320 → 256
        #

        self.fusion = nn.Sequential(

            nn.Conv2d(
                in_channels=256 + 64,
                out_channels=256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),
        )

        # ==========================================================
        # 4. Final BEV feature projection
        # ==========================================================

        self.output_block = nn.Sequential(

            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=128,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True),
        )

        # Initialisation des poids
        self._initialize_weights()

    # ==============================================================
    # Construction d'un stage ResNet
    # ==============================================================

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:

        blocks = []

        # Premier bloc:
        #
        # peut modifier:
        # - la résolution
        # - le nombre de channels

        blocks.append(
            BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
            )
        )

        # Blocs suivants:
        #
        # même résolution
        # même nombre de channels

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

    # ==============================================================
    # Initialisation
    # ==============================================================

    def _initialize_weights(self):

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
                nn.BatchNorm2d,
            ):

                nn.init.constant_(
                    module.weight,
                    1.0,
                )

                nn.init.constant_(
                    module.bias,
                    0.0,
                )

        # Comme ResNet zero_init_residual:
        #
        # chaque residual block commence
        # approximativement comme une identity.
        #

        for module in self.modules():

            if isinstance(
                module,
                BasicBlock,
            ):

                nn.init.constant_(
                    module.bn2.weight,
                    0.0,
                )

    # ==============================================================
    # Forward
    # ==============================================================

    def forward(
        self,
        bev: torch.Tensor,
    ) -> torch.Tensor:

        # ----------------------------------------------------------
        # Vérifications
        # ----------------------------------------------------------

        if bev.ndim != 4:

            raise ValueError(
                "BEVBackbone expects "
                "[B,C,Nx,Ny]. "
                f"Received: {tuple(bev.shape)}"
            )

        B, C, Nx, Ny = bev.shape

        if C != self.in_channels:

            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"received {C}."
            )

        # On conserve la taille originale.
        #
        # Ça nous permet de toujours revenir exactement
        # à la résolution d'entrée.

        input_size = bev.shape[-2:]

        # ==========================================================
        # 1. Stem
        # ==========================================================
        #
        # [B,64,200,200]
        #
        #       ↓
        #
        # [B,64,100,100]
        #

        x = self.stem(bev)

        # ==========================================================
        # 2. ResNet layer1
        # ==========================================================

        x1 = self.layer1(x)

        # x1:
        #
        # [B,64,100,100]

        # ==========================================================
        # 3. layer2
        # ==========================================================

        x2 = self.layer2(x1)

        # x2:
        #
        # [B,128,50,50]

        # ==========================================================
        # 4. layer3
        # ==========================================================

        x3 = self.layer3(x2)

        # x3:
        #
        # [B,256,25,25]

        # ==========================================================
        # 5. Upsample deep features vers x1
        # ==========================================================
        #
        # Au lieu de coder explicitement:
        #
        # scale_factor=4
        #
        # on demande exactement la taille x1.
        #
        # C'est plus robuste si plus tard Nx/Ny changent.
        #

        x3_up = F.interpolate(
            x3,
            size=x1.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

        # x3_up:
        #
        # [B,256,100,100]

        # ==========================================================
        # 6. Skip connection
        # ==========================================================

        x = torch.cat(
            [
                x1,
                x3_up,
            ],
            dim=1,
        )

        # x:
        #
        # [B,320,100,100]

        # ==========================================================
        # 7. Fusion
        # ==========================================================

        x = self.fusion(x)

        # x:
        #
        # [B,256,100,100]

        # ==========================================================
        # 8. Retour à la résolution BEV originale
        # ==========================================================

        x = F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=True,
        )

        # x:
        #
        # [B,256,200,200]

        # ==========================================================
        # 9. Output features
        # ==========================================================

        x = self.output_block(x)

        # Final:
        #
        # [B,128,200,200]

        return x


# ==============================================================
# Test manuel
# ==============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = BEVBackbone(
        in_channels=64,
        out_channels=128,
    ).to(device)

    bev = torch.randn(
        2,
        64,
        200,
        200,
        device=device,
    )

    print(
        "Input:",
        bev.shape,
    )

    with torch.no_grad():

        output = model(bev)

    print(
        "Output:",
        output.shape,
    )

    assert output.shape == (
        2,
        128,
        200,
        200,
    )

    print(
        "BEVBackbone test PASSED"
    )