#!/usr/bin/env python3

from typing import Dict

import torch
import torch.nn as nn


"""
Un détail important : le Conv1×1 est le seul élément neuronal
spécifique du Lift ici. Le softmax, les unsqueeze,
la multiplication et le permute sont des opérations
déterministes. C’est donc principalement cette convolution
qui apprend à transformer les 512 features du CameraEncoder
en distribution de profondeur + contexte BEV transportable.
"""


class Lift(nn.Module):
    """
    Lift module.

    Transforme une feature map image 2D en un volume de features
    distribué suivant plusieurs hypothèses de profondeur.

    ------------------------------------------------------------------
    INPUT
    ------------------------------------------------------------------

    image_features:
        [B, N, Cin, Hf, Wf]

    Exemple:
        [2, 4, 512, 8, 22]


    ------------------------------------------------------------------
    OUTPUTS
    ------------------------------------------------------------------

    depth_logits:
        [B, N, D, Hf, Wf]

    depth_probs:
        [B, N, D, Hf, Wf]

    context:
        [B, N, C, Hf, Wf]

    lifted_features:
        [B, N, D, Hf, Wf, C]


    Avec notre configuration:

        Cin = 512
        D   = 41
        C   = 64
    """

    def __init__(
        self,
        in_channels: int = 512,
        num_depth_bins: int = 41,
        context_channels: int = 64,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.D = num_depth_bins
        self.C = context_channels

        # --------------------------------------------------------------
        # Projection Depth + Context
        # --------------------------------------------------------------
        #
        # Pour chaque position spatiale (v, u):
        #
        # INPUT:
        #   512 features
        #
        # OUTPUT:
        #   D + C features
        #
        # Dans notre cas:
        #
        #   512
        #     ↓
        #   41 + 64
        #     ↓
        #   105 channels
        #
        # Les:
        #   - 41 premiers channels = depth logits
        #   - 64 derniers channels = context features
        #
        # kernel_size=1:
        # aucune modification de Hf/Wf.
        #
        self.depth_context_head = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.D + self.C,
            kernel_size=1,
            bias=True,
        )

    def forward(
        self,
        image_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        # --------------------------------------------------------------
        # 0. Vérification input
        # --------------------------------------------------------------

        if image_features.ndim != 5:
            raise ValueError(
                "Lift expects input shape [B, N, C, H, W]. "
                f"Received: {tuple(image_features.shape)}"
            )

        B, N, Cin, Hf, Wf = image_features.shape

        if Cin != self.in_channels:
            raise ValueError(
                f"Lift expects {self.in_channels} input channels, "
                f"but received {Cin}."
            )

        # --------------------------------------------------------------
        # 1. Fusion temporaire Batch + Cameras
        # --------------------------------------------------------------
        #
        # [B, N, 512, Hf, Wf]
        #
        #       ↓
        #
        # [B*N, 512, Hf, Wf]
        #
        # Le Lift est partagé entre toutes les caméras exactement
        # comme le CameraEncoder.
        #
        x = image_features.reshape(
            B * N,
            Cin,
            Hf,
            Wf,
        )

        # --------------------------------------------------------------
        # 2. Depth + Context prediction
        # --------------------------------------------------------------
        #
        # [B*N, 512, Hf, Wf]
        #
        #        ↓ Conv 1x1
        #
        # [B*N, D+C, Hf, Wf]
        #
        # Avec D=41 et C=64:
        #
        # [B*N, 105, Hf, Wf]
        #
        depth_context = self.depth_context_head(x)

        # --------------------------------------------------------------
        # 3. Séparation Depth / Context
        # --------------------------------------------------------------

        # Les D premiers channels:
        #
        # [B*N, D, Hf, Wf]
        #
        depth_logits_flat = depth_context[:, :self.D]

        # Les C channels suivants:
        #
        # [B*N, C, Hf, Wf]
        #
        context_flat = depth_context[
            :,
            self.D:self.D + self.C,
        ]

        # --------------------------------------------------------------
        # 4. Conversion des depth logits en probabilités
        # --------------------------------------------------------------
        #
        # IMPORTANT:
        #
        # Ici la shape est:
        #
        # [B*N, D, Hf, Wf]
        #
        # donc l'axe profondeur est dim=1.
        #
        # Pour chaque pixel feature (v,u):
        #
        # sum_d P(d | u,v) = 1
        #
        depth_probs_flat = torch.softmax(
            depth_logits_flat,
            dim=1,
        )

        # --------------------------------------------------------------
        # 5. LIFT = Depth × Context
        # --------------------------------------------------------------
        #
        # DEPTH:
        #
        # [B*N, D, Hf, Wf]
        #
        # CONTEXT:
        #
        # [B*N, C, Hf, Wf]
        #
        # On veut:
        #
        # [B*N, C, D, Hf, Wf]
        #
        #
        # depth_probs_flat.unsqueeze(1)
        #
        # [B*N, 1, D, Hf, Wf]
        #
        #
        # context_flat.unsqueeze(2)
        #
        # [B*N, C, 1, Hf, Wf]
        #
        #
        # Broadcasting:
        #
        # [B*N, 1, D, Hf, Wf]
        #               ×
        # [B*N, C, 1, Hf, Wf]
        #
        #               ↓
        #
        # [B*N, C, D, Hf, Wf]
        #
        lifted_flat = (
            depth_probs_flat.unsqueeze(1)
            *
            context_flat.unsqueeze(2)
        )

        # --------------------------------------------------------------
        # 6. Restaurer B et N
        # --------------------------------------------------------------

        depth_logits = depth_logits_flat.reshape(
            B,
            N,
            self.D,
            Hf,
            Wf,
        )

        depth_probs = depth_probs_flat.reshape(
            B,
            N,
            self.D,
            Hf,
            Wf,
        )

        context = context_flat.reshape(
            B,
            N,
            self.C,
            Hf,
            Wf,
        )

        # lifted_flat est actuellement:
        #
        # [B*N, C, D, Hf, Wf]
        #
        lifted_features = lifted_flat.reshape(
            B,
            N,
            self.C,
            self.D,
            Hf,
            Wf,
        )

        # --------------------------------------------------------------
        # 7. Réorganisation pour notre convention
        # --------------------------------------------------------------
        #
        # Avant:
        #
        # [B, N, C, D, Hf, Wf]
        #
        # Nous voulons:
        #
        # [B, N, D, Hf, Wf, C]
        #
        # afin d'avoir:
        #
        # lifted_features[b,n,d,v,u]
        #
        #     → un vecteur de C=64 features.
        #
        lifted_features = lifted_features.permute(
            0,  # B
            1,  # N
            3,  # D
            4,  # Hf
            5,  # Wf
            2,  # C
        ).contiguous()

        # --------------------------------------------------------------
        # 8. Retour
        # --------------------------------------------------------------
        #
        # On garde volontairement toutes les valeurs intermédiaires.
        #
        # C'est utile:
        #   - pour apprendre,
        #   - pour debugger,
        #   - pour visualiser les depth distributions.
        #
        return {
            "depth_logits": depth_logits,
            "depth_probs": depth_probs,
            "context": context,
            "lifted_features": lifted_features,
        }


# ----------------------------------------------------------------------
# Test manuel rapide
# ----------------------------------------------------------------------

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    lift = Lift(
        in_channels=512,
        num_depth_bins=41,
        context_channels=64,
    ).to(device)

    # Simulation de la sortie CameraEncoder:
    #
    # B = 2
    # N = 4
    #
    image_features = torch.randn(
        2,
        4,
        512,
        8,
        22,
        device=device,
    )

    with torch.no_grad():
        outputs = lift(image_features)

    print(
        "Input          :",
        image_features.shape,
    )

    print(
        "Depth logits   :",
        outputs["depth_logits"].shape,
    )

    print(
        "Depth probs    :",
        outputs["depth_probs"].shape,
    )

    print(
        "Context        :",
        outputs["context"].shape,
    )

    print(
        "Lifted features:",
        outputs["lifted_features"].shape,
    )

    # Vérifier la somme des probabilités de profondeur.
    depth_sum = outputs["depth_probs"].sum(dim=2)

    print(
        "Depth sum min/max:",
        depth_sum.min().item(),
        depth_sum.max().item(),
    )

    assert outputs["depth_logits"].shape == (
        2, 4, 41, 8, 22
    )

    assert outputs["depth_probs"].shape == (
        2, 4, 41, 8, 22
    )

    assert outputs["context"].shape == (
        2, 4, 64, 8, 22
    )

    assert outputs["lifted_features"].shape == (
        2, 4, 41, 8, 22, 64
    )

    assert torch.allclose(
        depth_sum,
        torch.ones_like(depth_sum),
        atol=1e-5,
    )

    print("Lift test PASSED")