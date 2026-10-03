#!/usr/bin/env python3

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lss_det.data.nuscenes_dataset import (
    NuScenesLSSDataset,
)

from lss_det.data.collate import (
    lss_collate_fn,
)

from lss_det.models.lss_detector import (
    LSSDetector,
)

from lss_det.targets.centerpoint_targets import (
    CenterPointTargetBuilder,
)

from lss_det.losses.detection_loss import (
    DetectionLoss,
)

from lss_det.decoding.decode_boxes import (
    CenterPointDecoder,
)


# ==============================================================
# CONFIGURATION
# ==============================================================
#
# IMPORTANT:
#
# Ce script N'EST PAS encore notre vrai training nuScenes.
#
# Son objectif est uniquement de vérifier que le réseau peut
# mémoriser UNE SEULE frame.
#
# Si le modèle n'arrive pas à overfit une frame, il ne sert à rien
# de lancer un training complet.
#
# ==============================================================


DATAROOT = os.path.expanduser(
    "~/datasets/nuscenes"
)

VERSION = "v1.0-mini"

CAMERA_PRETRAINED = True
# ==============================================================
# SANITY OVERFIT
# ==============================================================
#
# On utilise UNE seule frame.
#
# Le réseau doit finir par quasiment la mémoriser.
#
# ==============================================================

NUM_SAMPLES = 10

BATCH_SIZE = 1

EPOCHS = 300


# ==============================================================
# OPTIMIZATION
# ==============================================================

LEARNING_RATE = 1e-4

NUM_WORKERS = 0

SEED = 42


# ==============================================================
# CAMERA ENCODER
# ==============================================================
#
# False pour notre premier test:
#
#   EfficientNet démarre également aléatoirement.
#
# C'est volontaire:
#
# on veut vérifier que toute l'architecture peut apprendre.
#
# Plus tard, pour le vrai training:
#
#   CAMERA_PRETRAINED = True
#
# sera fortement recommandé.
#
# ==============================================================

CAMERA_PRETRAINED = True


# ==============================================================
# DEBUG FREQUENCY
# ==============================================================
#
# Tous les N epochs, on inspecte:
#
# - heatmap aux centres GT
# - BEV
# - décodage
#
# ==============================================================

DEBUG_EVERY = 20


# ==============================================================
# CHECKPOINTS
# ==============================================================

CHECKPOINT_DIR = Path(
    "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "overfit_10_frames_groupnorm_best.pt"
)


# ==============================================================
# REPRODUCTIBILITY
# ==============================================================

def seed_everything(
    seed: int,
):
    """
    Rend le test aussi reproductible que possible.
    """

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ==============================================================
# MOVE TARGETS TO DEVICE
# ==============================================================

def move_targets_to_device(
    targets,
    device,
):
    """
    Le TargetBuilder travaille actuellement avec les GT CPU.

    On déplace ensuite tous les targets sur GPU.
    """

    return {
        key: value.to(
            device,
            non_blocking=True,
        )

        for key, value
        in targets.items()
    }


# ==============================================================
# MOVE MODEL INPUTS TO DEVICE
# ==============================================================

def get_model_inputs(
    batch,
    device,
):
    """
    Prépare les six entrées attendues par LSSDetector.

    Returns
    -------
    images:
        [B,N,3,H,W]

    intrins:
        [B,N,3,3]

    rots:
        [B,N,3,3]

    trans:
        [B,N,3]

    post_rots:
        [B,N,3,3]

    post_trans:
        [B,N,3]
    """

    return (
        batch["images"].to(
            device,
            non_blocking=True,
        ),

        batch["intrins"].to(
            device,
            non_blocking=True,
        ),

        batch["rots"].to(
            device,
            non_blocking=True,
        ),

        batch["trans"].to(
            device,
            non_blocking=True,
        ),

        batch["post_rots"].to(
            device,
            non_blocking=True,
        ),

        batch["post_trans"].to(
            device,
            non_blocking=True,
        ),
    )


# ==============================================================
# HEATMAP GT INSPECTION
# ==============================================================

@torch.no_grad()
def inspect_heatmap_targets(
    predictions,
    targets,
):
    """
    Inspecte la probabilité prédite EXACTEMENT aux centres GT.

    C'est beaucoup plus informatif que simplement:

        heatmap.max()

    car un maximum élevé ailleurs dans la carte peut être
    complètement faux.

    On veut idéalement voir pendant l'overfit:

        mean GT-center probability:
            0.10
              ↓
            0.30
              ↓
            0.70
              ↓
            0.90+
    """

    probabilities = torch.sigmoid(
        predictions["heatmap"]
    )

    B, K, Nx, Ny = (
        probabilities.shape
    )

    positive_probs = []

    # ==========================================================
    # Chaque sample du batch
    # ==========================================================

    for b in range(B):

        mask_b = targets[
            "mask"
        ][b]

        valid_indices = targets[
            "indices"
        ][
            b,
            mask_b,
        ]

        valid_labels = targets[
            "labels"
        ][
            b,
            mask_b,
        ]

        # ======================================================
        # Chaque objet GT
        # ======================================================

        for (
            index,
            label,
        ) in zip(
            valid_indices,
            valid_labels,
        ):

            linear_index = int(
                index.item()
            )

            class_id = int(
                label.item()
            )

            # --------------------------------------------------
            # Notre convention:
            #
            # linear_index =
            #
            # ix * Ny + iy
            #
            # --------------------------------------------------

            ix = (
                linear_index
                // Ny
            )

            iy = (
                linear_index
                % Ny
            )

            positive_probs.append(
                probabilities[
                    b,
                    class_id,
                    ix,
                    iy,
                ]
            )

    # ==========================================================
    # Aucun objet valide
    # ==========================================================

    if len(
        positive_probs
    ) == 0:

        print(
            "    No positive GT targets."
        )

        return

    positive_probs = torch.stack(
        positive_probs
    )

    print(
        "    GT-center probabilities:"
    )

    print(
        f"      count : "
        f"{positive_probs.numel()}"
    )

    print(
        f"      min   : "
        f"{positive_probs.min().item():.4f}"
    )

    print(
        f"      mean  : "
        f"{positive_probs.mean().item():.4f}"
    )

    print(
        f"      max   : "
        f"{positive_probs.max().item():.4f}"
    )

    print(
        f"    Global heatmap max: "
        f"{probabilities.max().item():.4f}"
    )


# ==============================================================
# BEV INSPECTION
# ==============================================================

@torch.no_grad()
def inspect_bev(
    bev_raw,
):
    """
    Vérifie que le VoxelPooling produit réellement une carte BEV
    contenant de l'information.

    bev_raw:
        [B,C,Nx,Ny]

    On inspecte notamment la proportion de cellules dans lesquelles
    au moins une feature non nulle a été déposée.
    """

    abs_bev = bev_raw.abs()

    # ==========================================================
    # Cellule considérée occupée si au moins un channel
    # contient une valeur significativement non nulle.
    # ==========================================================

    occupied = (
        abs_bev.sum(
            dim=1
        )
        > 1e-8
    )

    occupied_ratio = (
        occupied
        .float()
        .mean()
        .item()
    )

    print(
        "    BEV raw:"
    )

    print(
        f"      abs mean      : "
        f"{abs_bev.mean().item():.8f}"
    )

    print(
        f"      abs max       : "
        f"{abs_bev.max().item():.8f}"
    )

    print(
        f"      occupied ratio: "
        f"{occupied_ratio:.4f}"
    )


# ==============================================================
# HEATMAP GRADIENT INSPECTION
# ==============================================================

def inspect_heatmap_gradient(
    model,
):
    """
    Vérifie que la dernière convolution de la heatmap head
    reçoit réellement du gradient.
    """

    grad = (
        model
        .detection_head
        .heatmap_head
        .net[-1]
        .weight
        .grad
    )

    if grad is None:

        print(
            "    Heatmap final conv grad: NONE"
        )

        return

    print(
        "    Heatmap final conv grad:"
    )

    print(
        f"      abs mean: "
        f"{grad.abs().mean().item():.8f}"
    )

    print(
        f"      abs max : "
        f"{grad.abs().max().item():.8f}"
    )


# ==============================================================
# FIXED SAMPLE INSPECTION
# ==============================================================

@torch.no_grad()
def inspect_prediction(
    model,
    dataset,
    target_builder,
    decoder,
    device,
    sample_index=0,
):
    """
    Lance toujours exactement la même frame dans le réseau.

    Cela permet de suivre visuellement / numériquement
    l'évolution de l'overfit.
    """

    model.eval()

    sample = dataset[
        sample_index
    ]

    # ==========================================================
    # Ajouter artificiellement batch dimension B=1
    # ==========================================================

    images = (
        sample["images"]
        .unsqueeze(0)
        .to(device)
    )

    intrins = (
        sample["intrins"]
        .unsqueeze(0)
        .to(device)
    )

    rots = (
        sample["rots"]
        .unsqueeze(0)
        .to(device)
    )

    trans = (
        sample["trans"]
        .unsqueeze(0)
        .to(device)
    )

    post_rots = (
        sample["post_rots"]
        .unsqueeze(0)
        .to(device)
    )

    post_trans = (
        sample["post_trans"]
        .unsqueeze(0)
        .to(device)
    )

    # ==========================================================
    # Targets
    # ==========================================================

    targets = target_builder(
        gt_boxes=[
            sample["gt_boxes"]
        ],

        gt_labels=[
            sample["gt_labels"]
        ],
    )

    targets = move_targets_to_device(
        targets,
        device,
    )

    # ==========================================================
    # Forward
    # ==========================================================
    #
    # return_intermediates=True dans notre modèle.
    #
    # Donc output:
    #
    # {
    #     predictions,
    #     camera_features,
    #     depth_probs,
    #     geometry,
    #     bev_raw,
    #     ...
    # }
    #
    # ==========================================================

    outputs = model(
        images,
        intrins,
        rots,
        trans,
        post_rots,
        post_trans,
    )

    predictions = outputs[
        "predictions"
    ]

    # ==========================================================
    # GT information
    # ==========================================================

    print(
        f"    GT boxes dataset: "
        f"{sample['gt_boxes'].shape[0]}"
    )

    print(
        f"    GT targets in BEV: "
        f"{targets['mask'].sum().item()}"
    )

    # ==========================================================
    # Heatmap
    # ==========================================================

    inspect_heatmap_targets(
        predictions,
        targets,
    )

    # ==========================================================
    # BEV
    # ==========================================================

    inspect_bev(
        outputs[
            "bev_raw"
        ]
    )

    # ==========================================================
    # Decoder
    # ==========================================================

    detections = decoder(
        predictions
    )[0]

    print(
        f"    decoded boxes: "
        f"{detections['boxes'].shape[0]}"
    )

    # ==========================================================
    # Show first predictions
    # ==========================================================

    if (
        detections[
            "boxes"
        ].shape[0]
        > 0
    ):

        num_show = min(
            5,
            detections[
                "boxes"
            ].shape[0],
        )

        print()
        print(
            "    Predictions:"
        )

        for i in range(
            num_show
        ):

            score = float(
                detections[
                    "scores"
                ][i]
                .item()
            )

            label = int(
                detections[
                    "labels"
                ][i]
                .item()
            )

            box = (
                detections[
                    "boxes"
                ][i]
                .detach()
                .cpu()
            )

            print(
                f"      #{i} "
                f"class={label} "
                f"score={score:.3f} "
                f"box={box.tolist()}"
            )

    # ==========================================================
    # Restore training mode
    # ==========================================================

    model.train()


# ==============================================================
# MAIN
# ==============================================================

def main():

    seed_everything(
        SEED
    )

    # ==========================================================
    # DEVICE
    # ==========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "======================================"
    )

    print(
        "LSS DETECTOR - ONE FRAME OVERFIT"
    )

    print(
        "======================================"
    )

    print(
        "Device:",
        device,
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    print(
        "Dataset:",
        DATAROOT,
    )

    print(
        "Samples used:",
        NUM_SAMPLES,
    )

    # ==========================================================
    # DATASET
    # ==========================================================

    full_dataset = (
        NuScenesLSSDataset(
            dataroot=DATAROOT,

            version=VERSION,

            split="train",

            verbose=False,
        )
    )

    print(
        "Full train samples:",
        len(
            full_dataset
        ),
    )

    print(
        "Classes:",
        full_dataset.classes,
    )

    print(
        "Cameras:",
        full_dataset.cameras,
    )

    # ==========================================================
    # OVERFIT SUBSET
    # ==========================================================
    #
    # On prend volontairement les premières samples.
    #
    # Ici:
    #
    # NUM_SAMPLES = 1
    #
    # donc toujours exactement la même frame.
    #
    # ==========================================================

    num_samples = min(
        NUM_SAMPLES,
        len(
            full_dataset
        ),
    )

    subset_indices = list(
        range(
            num_samples
        )
    )

    train_dataset = Subset(
        full_dataset,
        subset_indices,
    )

    # ==========================================================
    # DATALOADER
    # ==========================================================
    #
    # shuffle=False:
    #
    # volontaire pour avoir un test totalement déterministe.
    #
    # ==============================================================

    train_loader = DataLoader(
        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        collate_fn=lss_collate_fn,

        pin_memory=(
            device.type
            == "cuda"
        ),

        drop_last=False,
    )

    # ==========================================================
    # TARGET BUILDER
    # ==========================================================

    target_builder = (
        CenterPointTargetBuilder(

            classes=(
                full_dataset.classes
            ),

            xbound=(
                -50.0,
                50.0,
                0.5,
            ),

            ybound=(
                -50.0,
                50.0,
                0.5,
            ),

            max_objects=500,

            gaussian_overlap=0.1,

            min_radius=2,
        )
    )

    # ==========================================================
    # MODEL
    # ==========================================================
    #
    # IMPORTANT:
    #
    # return_intermediates=True
    #
    # afin de pouvoir inspecter:
    #
    #   depth
    #   geometry
    #   raw BEV
    #   BEV features
    #
    # pendant ce sanity-check.
    #
    # ==============================================================

    model = LSSDetector(

        num_classes=len(
            full_dataset.classes
        ),

        camera_pretrained=(
            CAMERA_PRETRAINED
        ),

        return_intermediates=True,
    )

    model = model.to(
        device
    )

    # ==========================================================
    # LOSS
    # ==========================================================

    criterion = DetectionLoss(

        heatmap_weight=1.0,

        reg_weight=1.0,

        center_z_weight=1.0,

        dim_weight=1.0,

        rot_weight=1.0,
    )

    # ==========================================================
    # DECODER
    # ==========================================================
    #
    # Utilisé uniquement pour le debug / inference.
    #
    # Il n'est PAS utilisé dans backward().
    #
    # ==============================================================

    decoder = CenterPointDecoder(

        xbound=(
            -50.0,
            50.0,
            0.5,
        ),

        ybound=(
            -50.0,
            50.0,
            0.5,
        ),

        top_k=100,

        # ------------------------------------------------------
        # Très bas pendant l'overfit.
        #
        # On veut voir les prédictions évoluer même lorsqu'elles
        # sont encore faibles.
        # ------------------------------------------------------

        score_threshold=0.1,

        use_circle_nms=True,

        nms_min_distance=1.0,
    )

    # ==========================================================
    # OPTIMIZER
    # ==========================================================
    #
    # Pas de weight decay.
    #
    # Ici on VEUT explicitement que le modèle mémorise.
    #
    # ==============================================================

    optimizer = torch.optim.Adam(
        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=0.0,
    )

    # ==========================================================
    # MODEL SIZE
    # ==========================================================

    total_parameters = sum(
        parameter.numel()

        for parameter
        in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()

        for parameter
        in model.parameters()

        if parameter.requires_grad
    )

    print(
        f"Parameters: "
        f"{total_parameters / 1e6:.2f} M"
    )

    print(
        f"Trainable: "
        f"{trainable_parameters / 1e6:.2f} M"
    )

    # ==========================================================
    # INITIAL GT INSPECTION
    # ==========================================================

    first_sample = full_dataset[
        0
    ]

    initial_targets = target_builder(

        gt_boxes=[
            first_sample[
                "gt_boxes"
            ]
        ],

        gt_labels=[
            first_sample[
                "gt_labels"
            ]
        ],
    )

    print()
    print(
        "======================================"
    )

    print(
        "FIRST SAMPLE"
    )

    print(
        "======================================"
    )

    print(
        "GT boxes dataset:",
        first_sample[
            "gt_boxes"
        ].shape[0],
    )

    print(
        "GT targets inside BEV:",
        initial_targets[
            "mask"
        ].sum().item(),
    )

    print(
        "Sample token:",
        first_sample[
            "sample_token"
        ],
    )

    print(
        "======================================"
    )

    print()

    # ==========================================================
    # TRAINING
    # ==========================================================

    best_loss = float(
        "inf"
    )

    model.train()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        # ======================================================
        # EPOCH ACCUMULATORS
        # ======================================================

        epoch_total = 0.0

        epoch_heatmap = 0.0

        epoch_reg = 0.0

        epoch_z = 0.0

        epoch_dim = 0.0

        epoch_rot = 0.0

        num_batches = 0

        # Used later for debug.
        last_predictions = None
        last_targets = None
        last_outputs = None

        # ======================================================
        # TRAINING BATCHES
        # ======================================================

        for batch in train_loader:

            # ==================================================
            # 1. BUILD TARGETS
            # ==================================================
            #
            # gt_boxes:
            #
            # List[Tensor[M_i,7]]
            #
            # parce que chaque frame possède un nombre différent
            # de GT.
            #
            # ==================================================

            targets = target_builder(

                gt_boxes=batch[
                    "gt_boxes"
                ],

                gt_labels=batch[
                    "gt_labels"
                ],
            )

            targets = move_targets_to_device(
                targets,
                device,
            )

            # ==================================================
            # 2. MODEL INPUTS
            # ==================================================

            (
                images,
                intrins,
                rots,
                trans,
                post_rots,
                post_trans,
            ) = get_model_inputs(
                batch,
                device,
            )

            # ==================================================
            # 3. FORWARD
            # ==================================================
            #
            # return_intermediates=True
            #
            # Donc:
            #
            # outputs["predictions"]
            #
            # contient la vraie sortie de la DetectionHead.
            #
            # ==================================================

            outputs = model(

                images,

                intrins,

                rots,

                trans,

                post_rots,

                post_trans,
            )

            predictions = outputs[
                "predictions"
            ]

            # ==================================================
            # 4. LOSS
            # ==================================================

            losses = criterion(
                predictions,
                targets,
            )

            loss = losses[
                "loss"
            ]

            # ==================================================
            # 5. SAFETY
            # ==================================================

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    f"Non-finite loss "
                    f"at epoch {epoch}: "
                    f"{loss.item()}"
                )

            # ==================================================
            # 6. ZERO GRADIENT
            # ==================================================

            optimizer.zero_grad(
                set_to_none=True
            )

            # ==================================================
            # 7. BACKWARD
            # ==================================================

            loss.backward()

            # ==================================================
            # 8. GRADIENT DEBUG
            # ==================================================
            #
            # On inspecte particulièrement la heatmap head.
            #
            # ==================================================

            if epoch in (
                1,
                10,
                20,
                50,
                100,
                200,
                300,
            ):

                print()
                print(
                    f"  [gradient debug epoch {epoch}]"
                )

                inspect_heatmap_gradient(
                    model
                )

            # ==================================================
            # 9. GRADIENT CLIPPING
            # ==========================================================

            total_grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=10.0,
                )
            )

            # ==================================================
            # 10. OPTIMIZER STEP
            # ==========================================================

            optimizer.step()

            # ==================================================
            # 11. STATISTICS
            # ==========================================================

            epoch_total += float(
                losses[
                    "loss"
                ]
                .detach()
                .item()
            )

            epoch_heatmap += float(
                losses[
                    "loss_heatmap"
                ]
                .detach()
                .item()
            )

            epoch_reg += float(
                losses[
                    "loss_reg"
                ]
                .detach()
                .item()
            )

            epoch_z += float(
                losses[
                    "loss_center_z"
                ]
                .detach()
                .item()
            )

            epoch_dim += float(
                losses[
                    "loss_dim"
                ]
                .detach()
                .item()
            )

            epoch_rot += float(
                losses[
                    "loss_rot"
                ]
                .detach()
                .item()
            )

            num_batches += 1

            # Garder le dernier batch pour debug.
            last_predictions = predictions

            last_targets = targets

            last_outputs = outputs

        # ======================================================
        # EPOCH AVERAGES
        # ======================================================

        avg_total = (
            epoch_total
            / num_batches
        )

        avg_heatmap = (
            epoch_heatmap
            / num_batches
        )

        avg_reg = (
            epoch_reg
            / num_batches
        )

        avg_z = (
            epoch_z
            / num_batches
        )

        avg_dim = (
            epoch_dim
            / num_batches
        )

        avg_rot = (
            epoch_rot
            / num_batches
        )

        # ======================================================
        # LOG
        # ======================================================

        print(
            f"Epoch "
            f"{epoch:03d}/{EPOCHS} | "

            f"total={avg_total:.4f} | "

            f"hm={avg_heatmap:.4f} | "

            f"reg={avg_reg:.4f} | "

            f"z={avg_z:.4f} | "

            f"dim={avg_dim:.4f} | "

            f"rot={avg_rot:.4f}"
        )

        # ======================================================
        # SAVE BEST CHECKPOINT
        # ======================================================

        if avg_total < best_loss:

            best_loss = (
                avg_total
            )

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "loss":
                        best_loss,

                    "classes":
                        full_dataset.classes,

                    "num_samples":
                        NUM_SAMPLES,

                    "learning_rate":
                        LEARNING_RATE,
                },

                CHECKPOINT_PATH,
            )

        # ======================================================
        # LIGHT DEBUG USING CURRENT TRAINING BATCH
        # ======================================================
        #
        # Ici on peut inspecter directement la heatmap produite
        # pendant le dernier forward.
        #
        # ======================================================

        if (
            epoch == 1
            or
            epoch % DEBUG_EVERY == 0
        ):

            print()
            print(
                "  =================================="
            )

            print(
                f"  DEBUG EPOCH {epoch}"
            )

            print(
                "  =================================="
            )

            inspect_heatmap_targets(
                last_predictions,
                last_targets,
            )

            inspect_bev(
                last_outputs[
                    "bev_raw"
                ]
            )

            print(
                "  =================================="
            )

            # ==================================================
            # FIXED SAMPLE FULL DEBUG
            # ==================================================

            print()
            print(
                "  --- fixed sample debug ---"
            )

            inspect_prediction(

                model=model,

                dataset=full_dataset,

                target_builder=target_builder,

                decoder=decoder,

                device=device,

                sample_index=0,
            )

            print(
                "  --------------------------"
            )

            print()

    # ==========================================================
    # FINISHED
    # ==========================================================

    print()
    print(
        "======================================"
    )

    print(
        "ONE FRAME OVERFIT FINISHED"
    )

    print(
        "======================================"
    )

    print(
        "Best loss:",
        best_loss,
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )

    print(
        "======================================"
    )


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":

    main()