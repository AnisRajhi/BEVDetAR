#!/usr/bin/env python3

from typing import Tuple, Optional

import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet


class FeatureFusion(nn.Module):
    """
    Fusion des features multi-échelles d'EfficientNet-B0.

    Entrées
    -------
    reduction_4:
        [M, 112, H/16, W/16]

    reduction_5:
        [M, 320, H/32, W/32]

    où:
        M = B * N

    Sortie
    ------
    [M, 512, H/16, W/16]
    """

    def __init__(self):
        super().__init__()

        # reduction_5 est deux fois plus petite spatialement que reduction_4.
        #
        # Exemple:
        #   reduction_4 = [M, 112, 8, 22]
        #   reduction_5 = [M, 320, 4, 11]
        #
        # On remonte reduction_5 vers 8x22.
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        # UPSAMPLE : No trainable param ! → Unlike Transpose

        # Après upsample:
        #
        #   reduction_4          : 112 channels
        #   reduction_5 upsample : 320 channels
        #
        # concaténation:
        #
        #   112 + 320 = 432 channels
        #
        # Puis on transforme les 432 channels vers 512 channels.
    
        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_channels=112 + 320,
                out_channels=512,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        #Pourquoi les deux conv2 3x3:
        #La convolution permet d'apprendre :
        #    432 channels
        #        ↓
        #    combinaisons utiles
        #        ↓
        #    512 nouvelles features

    def forward(
        self,
        reduction_4: torch.Tensor,
        reduction_5: torch.Tensor,
    ) -> torch.Tensor:

        # --------------------------------------------------------------
        # 1. Upsample reduction_5
        # --------------------------------------------------------------
        #
        # [M, 320, H/32, W/32]
        #
        #       ↓
        #
        # [M, 320, H/16, W/16]
        #
        reduction_5_up = self.upsample(reduction_5)

        # Sécurité:
        # les dimensions spatiales doivent maintenant être identiques.
        if reduction_5_up.shape[-2:] != reduction_4.shape[-2:]:
            raise RuntimeError(
                "Spatial size mismatch during feature fusion:\n"
                f"  reduction_4    : {reduction_4.shape}\n"
                f"  reduction_5_up : {reduction_5_up.shape}"
            )

        # --------------------------------------------------------------
        # 2. Concaténation sur l'axe des channels
        # --------------------------------------------------------------
        #
        # reduction_4:
        # [M, 112, H/16, W/16]
        #
        # reduction_5_up:
        # [M, 320, H/16, W/16]
        #
        # résultat:
        # [M, 432, H/16, W/16]
        #
        x = torch.cat(
            [reduction_4, reduction_5_up],
            dim=1,
        )

        # --------------------------------------------------------------
        # 3. Fusion CNN
        # --------------------------------------------------------------
        #
        # [M, 432, H/16, W/16]
        #
        #        ↓
        #
        # [M, 512, H/16, W/16]
        #
        x = self.fusion(x)

        return x


class CameraEncoder(nn.Module):
    """
    Camera Encoder multi-caméra basé sur EfficientNet-B0.

    Le même réseau est partagé entre toutes les caméras.

    Input
    -----
    images:
        [B, N, 3, H, W]

    Output
    ------
    features:
        [B, N, 512, H/16, W/16]

    Exemple
    -------
    input:
        [2, 4, 3, 128, 352]

    output:
        [2, 4, 512, 8, 22]
    """

    def __init__(
        self,
        pretrained: bool = True,
        weights_path: Optional[str] = None,
    ):
        super().__init__()

        # --------------------------------------------------------------
        # EfficientNet-B0 backbone
        # --------------------------------------------------------------

        if pretrained:
            # Charge les poids ImageNet.
            #
            # Si weights_path est None:
            # efficientnet_pytorch utilise/télécharge les poids standards.
            #
            # Si weights_path est fourni:
            # on charge les poids depuis ce fichier.
            self.backbone = EfficientNet.from_pretrained(
                "efficientnet-b0",
                weights_path=weights_path,
            )

        else:
            # Même architecture mais poids aléatoires.
            self.backbone = EfficientNet.from_name(
                "efficientnet-b0"
            )

        # --------------------------------------------------------------
        # Fusion reduction_4 + reduction_5
        # --------------------------------------------------------------

        self.feature_fusion = FeatureFusion()

    def extract_backbone_features(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extrait les deux niveaux EfficientNet utilisés par LSS.

        Input
        -----
        images:
            [M, 3, H, W]

        Returns
        -------
        reduction_4:
            [M, 112, H/16, W/16]

        reduction_5:
            [M, 320, H/32, W/32]
        """

        # EfficientNet fournit directement plusieurs sorties
        # intermédiaires avec extract_endpoints().
        endpoints = self.backbone.extract_endpoints(images)

        reduction_4 = endpoints["reduction_4"]
        reduction_5 = endpoints["reduction_5"]

        return reduction_4, reduction_5

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:

        # --------------------------------------------------------------
        # Vérification de l'input
        # --------------------------------------------------------------

        if images.ndim != 5:
            raise ValueError(
                "CameraEncoder expects images with shape "
                "[B, N, 3, H, W]. "
                f"Received: {tuple(images.shape)}"
            )

        B, N, C, H, W = images.shape

        if C != 3:
            raise ValueError(
                "CameraEncoder expects RGB images (C=3). "
                f"Received C={C}"
            )

        # --------------------------------------------------------------
        # 1. Fusion temporaire Batch + Cameras
        # --------------------------------------------------------------
        #
        # Input:
        #
        # [B, N, 3, H, W]
        #
        # Exemple:
        #
        # [2, 4, 3, 128, 352]
        #
        # On veut faire passer toutes les images dans le MÊME
        # EfficientNet + C'est un NN classique qui attend [B, 3, H, W]
        # Il connait pas le concept de multi-cam
        #
        # Donc:
        #
        # [B, N, 3, H, W]
        #
        #       ↓ reshape
        #
        # [B*N, 3, H, W]
        #
        x = images.reshape(
            B * N,
            C,
            H,
            W,
        )

        # --------------------------------------------------------------
        # 2. EfficientNet
        # --------------------------------------------------------------

        reduction_4, reduction_5 = self.extract_backbone_features(x)

        # Pour H=128, W=352:
        #
        # reduction_4:
        # [B*N, 112, 8, 22]
        #
        # reduction_5:
        # [B*N, 320, 4, 11]

        # --------------------------------------------------------------
        # 3. Fusion multi-échelle
        # --------------------------------------------------------------

        features = self.feature_fusion(
            reduction_4,
            reduction_5,
        )

        # features:
        #
        # [B*N, 512, H/16, W/16]
        #
        # Exemple:
        #
        # [B*N, 512, 8, 22]

        # --------------------------------------------------------------
        # 4. Restaurer la dimension Camera N
        # --------------------------------------------------------------

        _, C_out, H_out, W_out = features.shape

        features = features.reshape(
            B,
            N,
            C_out,
            H_out,
            W_out,
        )

        # Final:
        #
        # [B, N, 512, H/16, W/16]

        return features


# ----------------------------------------------------------------------
# Test rapide lorsqu'on lance directement le fichier
# ----------------------------------------------------------------------

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # Pour éviter de télécharger les poids lors de ce simple test:
    encoder = CameraEncoder(
        pretrained=False
    ).to(device)

    # Exemple:
    #
    # B = 2 samples
    # N = 4 cameras
    # RGB
    # H = 128
    # W = 352
    images = torch.randn(
        2,
        4,
        3,
        128,
        352,
        device=device,
    )

    print(f"Input shape : {images.shape}")

    with torch.no_grad():
        features = encoder(images)

    print(f"Output shape: {features.shape}")

    expected_shape = (
        2,
        4,
        512,
        8,
        22,
    )

    assert features.shape == expected_shape, (
        f"Expected {expected_shape}, "
        f"got {tuple(features.shape)}"
    )

    print("CameraEncoder test PASSED")