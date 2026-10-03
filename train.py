#!/usr/bin/env python3

import os
import random
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader

from lss_det.data.nuscenes_dataset import (
    NuScenesLSSDataset,
)

from lss_det.data.transforms import (
    LSSImageTransform,
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
# Expérience:
#
#   nuScenes mini
#
#   AdamW
#   +
#   weight decay
#   +
#   augmentation photométrique
#
#
# IMPORTANT:
#
# Les augmentations sont appliquées UNIQUEMENT au train.
#
# La validation reste totalement déterministe.
#
# ==============================================================


DATAROOT = os.path.expanduser(
    "~/datasets/nuscenes"
)

VERSION = "v1.0-mini"

TRAIN_SPLIT = "train"

VAL_SPLIT = "val"


# ==============================================================
# TRAINING
# ==============================================================

EPOCHS = 30

BATCH_SIZE = 1

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

SEED = 42


# ==============================================================
# CAMERA ENCODER
# ==============================================================

CAMERA_PRETRAINED = True


# ==============================================================
# DEBUG
# ==============================================================

DEBUG_EVERY = 5


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


BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "nuscenes_mini_adamw_photoaug_best_val.pt"
)


LAST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "nuscenes_mini_adamw_photoaug_last.pt"
)


# ==============================================================
# RANDOM SEEDS
# ==============================================================

def seed_everything(
    seed: int,
):
    """
    Rend l'expérience aussi reproductible que possible.

    Attention:

    ColorJitter reste volontairement aléatoire pendant
    l'entraînement.

    Avec une seed fixe, l'expérience reste néanmoins
    globalement reproductible.
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
    Déplace tous les tensors créés par le TargetBuilder
    vers le device utilisé par le réseau.
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
# MODEL INPUTS
# ==============================================================

def get_model_inputs(
    batch,
    device,
):
    """
    Prépare les six entrées nécessaires au LSSDetector.

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
# LOSS ACCUMULATOR
# ==============================================================

def create_loss_accumulator():
    """
    Initialise les accumulateurs des différentes losses.
    """

    return {

        "loss":
            0.0,

        "loss_heatmap":
            0.0,

        "loss_reg":
            0.0,

        "loss_center_z":
            0.0,

        "loss_dim":
            0.0,

        "loss_rot":
            0.0,
    }


def accumulate_losses(
    accumulator,
    losses,
):
    """
    Ajoute les losses du batch courant.
    """

    for key in accumulator:

        accumulator[
            key
        ] += float(

            losses[
                key
            ]
            .detach()
            .item()
        )


def average_losses(
    accumulator,
    count,
):
    """
    Moyenne des losses sur un epoch complet.
    """

    if count == 0:

        raise RuntimeError(
            "Cannot average zero batches."
        )

    return {

        key: value / count

        for key, value
        in accumulator.items()
    }


# ==============================================================
# GT CENTER HEATMAP PROBABILITY
# ==============================================================

@torch.no_grad()
def get_gt_center_probability(
    predictions,
    targets,
):
    """
    Calcule la probabilité heatmap moyenne exactement
    aux centres des objets GT.

    Ce n'est PAS une vraie métrique de détection.

    C'est seulement un outil de diagnostic.

    Exemple:

        GTp train ↑
        GTp val   ↑

    est généralement bon signe.

    Si:

        GTp train ↑
        GTp val   ↓

    le réseau est probablement en train d'overfit.
    """

    probabilities = torch.sigmoid(
        predictions[
            "heatmap"
        ]
    )

    B, K, Nx, Ny = (
        probabilities.shape
    )

    values = []

    # ==========================================================
    # Chaque sample du batch
    # ==========================================================

    for b in range(B):

        valid_indices = targets[
            "indices"
        ][
            b,
            targets[
                "mask"
            ][b],
        ]

        valid_labels = targets[
            "labels"
        ][
            b,
            targets[
                "mask"
            ][b],
        ]

        # ======================================================
        # Chaque objet
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
            #     ix * Ny + iy
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

            values.append(

                probabilities[
                    b,
                    class_id,
                    ix,
                    iy,
                ]
            )

    if len(
        values
    ) == 0:

        return None

    return (

        torch.stack(
            values
        )

        .mean()

        .item()
    )


# ==============================================================
# TRAIN ONE EPOCH
# ==============================================================

def train_one_epoch(
    model,
    loader,
    target_builder,
    criterion,
    optimizer,
    device,
):
    """
    Training complet d'un epoch.

    Pipeline:

        batch
          ↓
        targets
          ↓
        forward LSS
          ↓
        detection loss
          ↓
        backward
          ↓
        optimizer.step()
    """

    # ==========================================================
    # TRAIN MODE
    # ==========================================================
    #
    # Grâce à notre CameraEncoder.train():
    #
    # - EfficientNet convolutions restent trainables
    # - EfficientNet BatchNorm restent figées en eval
    #
    # Nos propres couches utilisent GroupNorm.
    #
    # ==========================================================

    model.train()

    accumulator = (
        create_loss_accumulator()
    )

    num_batches = 0

    gt_probability_sum = 0.0

    gt_probability_count = 0

    # ==========================================================
    # BATCH LOOP
    # ==========================================================

    for batch in loader:

        # ======================================================
        # 1. TARGET GENERATION
        # ======================================================

        targets = target_builder(

            gt_boxes=batch[
                "gt_boxes"
            ],

            gt_labels=batch[
                "gt_labels"
            ],
        )

        targets = (
            move_targets_to_device(
                targets,
                device,
            )
        )

        # ======================================================
        # 2. MODEL INPUTS
        # ======================================================

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

        # ======================================================
        # 3. FORWARD
        # ======================================================

        predictions = model(

            images,

            intrins,

            rots,

            trans,

            post_rots,

            post_trans,
        )

        # ======================================================
        # 4. LOSS
        # ======================================================

        losses = criterion(
            predictions,
            targets,
        )

        loss = losses[
            "loss"
        ]

        # ======================================================
        # SAFETY
        # ======================================================

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Non-finite training loss."
            )

        # ======================================================
        # 5. ZERO GRAD
        # ======================================================

        optimizer.zero_grad(
            set_to_none=True
        )

        # ======================================================
        # 6. BACKWARD
        # ======================================================

        loss.backward()

        # ======================================================
        # 7. GRADIENT CLIPPING
        # ======================================================

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=10.0,
        )

        # ======================================================
        # 8. OPTIMIZER STEP
        # ======================================================

        optimizer.step()

        # ======================================================
        # 9. STATISTICS
        # ======================================================

        accumulate_losses(
            accumulator,
            losses,
        )

        center_probability = (
            get_gt_center_probability(
                predictions,
                targets,
            )
        )

        if (
            center_probability
            is not None
        ):

            gt_probability_sum += (
                center_probability
            )

            gt_probability_count += 1

        num_batches += 1

    # ==========================================================
    # AVERAGES
    # ==========================================================

    average = average_losses(
        accumulator,
        num_batches,
    )

    if (
        gt_probability_count
        > 0
    ):

        average[
            "gt_center_probability"
        ] = (

            gt_probability_sum

            / gt_probability_count
        )

    else:

        average[
            "gt_center_probability"
        ] = float(
            "nan"
        )

    return average


# ==============================================================
# VALIDATION
# ==============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    target_builder,
    criterion,
    device,
):
    """
    Validation indépendante.

    IMPORTANT:

        - aucune augmentation aléatoire
        - model.eval()
        - torch.no_grad()
        - aucun backward
        - aucun optimizer.step()
    """

    model.eval()

    accumulator = (
        create_loss_accumulator()
    )

    num_batches = 0

    gt_probability_sum = 0.0

    gt_probability_count = 0

    # ==========================================================
    # VALIDATION LOOP
    # ==========================================================

    for batch in loader:

        # ======================================================
        # TARGETS
        # ======================================================

        targets = target_builder(

            gt_boxes=batch[
                "gt_boxes"
            ],

            gt_labels=batch[
                "gt_labels"
            ],
        )

        targets = (
            move_targets_to_device(
                targets,
                device,
            )
        )

        # ======================================================
        # INPUTS
        # ======================================================

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

        # ======================================================
        # FORWARD
        # ======================================================

        predictions = model(

            images,

            intrins,

            rots,

            trans,

            post_rots,

            post_trans,
        )

        # ======================================================
        # LOSS
        # ======================================================

        losses = criterion(
            predictions,
            targets,
        )

        loss = losses[
            "loss"
        ]

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Non-finite validation loss."
            )

        # ======================================================
        # STATISTICS
        # ======================================================

        accumulate_losses(
            accumulator,
            losses,
        )

        center_probability = (
            get_gt_center_probability(
                predictions,
                targets,
            )
        )

        if (
            center_probability
            is not None
        ):

            gt_probability_sum += (
                center_probability
            )

            gt_probability_count += 1

        num_batches += 1

    # ==========================================================
    # AVERAGES
    # ==========================================================

    average = average_losses(
        accumulator,
        num_batches,
    )

    if (
        gt_probability_count
        > 0
    ):

        average[
            "gt_center_probability"
        ] = (

            gt_probability_sum

            / gt_probability_count
        )

    else:

        average[
            "gt_center_probability"
        ] = float(
            "nan"
        )

    return average


# ==============================================================
# VALIDATION SAMPLE DEBUG
# ==============================================================

@torch.no_grad()
def inspect_validation_sample(
    model,
    dataset,
    decoder,
    device,
    sample_index=0,
):
    """
    Inspecte toujours la même frame de validation.

    Cela permet d'observer l'évolution qualitative
    des scores de détection pendant le training.

    Ce debug ne remplace PAS une vraie métrique
    de détection.
    """

    model.eval()

    sample = dataset[
        sample_index
    ]

    # ==========================================================
    # Add batch dimension
    # ==============================================================

    images = (
        sample[
            "images"
        ]
        .unsqueeze(0)
        .to(device)
    )

    intrins = (
        sample[
            "intrins"
        ]
        .unsqueeze(0)
        .to(device)
    )

    rots = (
        sample[
            "rots"
        ]
        .unsqueeze(0)
        .to(device)
    )

    trans = (
        sample[
            "trans"
        ]
        .unsqueeze(0)
        .to(device)
    )

    post_rots = (
        sample[
            "post_rots"
        ]
        .unsqueeze(0)
        .to(device)
    )

    post_trans = (
        sample[
            "post_trans"
        ]
        .unsqueeze(0)
        .to(device)
    )

    # ==========================================================
    # Forward
    # ==============================================================

    predictions = model(

        images,

        intrins,

        rots,

        trans,

        post_rots,

        post_trans,
    )

    # ==========================================================
    # Decode
    # ==============================================================

    detections = decoder(
        predictions
    )[0]

    print(
        "    Validation sample:"
    )

    print(
        f"      GT boxes        : "
        f"{sample['gt_boxes'].shape[0]}"
    )

    print(
        f"      decoded boxes   : "
        f"{detections['boxes'].shape[0]}"
    )

    # ==========================================================
    # Highest detection scores
    # ==============================================================

    if (
        detections[
            "scores"
        ].numel()
        > 0
    ):

        print(
            f"      best score      : "
            f"{detections['scores'].max().item():.4f}"
        )

        num_show = min(

            3,

            detections[
                "boxes"
            ].shape[0],
        )

        for i in range(
            num_show
        ):

            print(

                f"      #{i} "

                f"class="
                f"{detections['labels'][i].item()} "

                f"score="
                f"{detections['scores'][i].item():.3f}"
            )


# ==============================================================
# SAVE CHECKPOINT
# ==============================================================

def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    train_losses,
    val_losses,
    classes,
):
    """
    Sauvegarde l'état nécessaire pour reprendre ou analyser
    un entraînement.
    """

    torch.save(

        {

            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "train_losses":
                train_losses,

            "val_losses":
                val_losses,

            "classes":
                classes,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "photometric_augmentation":
                True,
        },

        path,
    )


# ==============================================================
# MAIN
# ==============================================================

def main():

    seed_everything(
        SEED
    )

    # ==========================================================
    # DEVICE
    # ==============================================================

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
        "LSS DETECTOR - NUSCENES MINI TRAINING"
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

    # ==========================================================
    # IMAGE TRANSFORMS
    # ==========================================================
    #
    # TRAIN:
    #
    #   photometric augmentation ON
    #
    #
    # VALIDATION:
    #
    #   photometric augmentation OFF
    #
    #
    # Geometry remains identical.
    #
    # ==============================================================

    train_transform = (
        LSSImageTransform(

            final_dim=(
                128,
                352,
            ),

            bottom_crop_pct=0.11,

            normalize=True,

            training=True,

            brightness=0.2,

            contrast=0.2,

            saturation=0.2,

            hue=0.05,
        )
    )

    val_transform = (
        LSSImageTransform(

            final_dim=(
                128,
                352,
            ),

            bottom_crop_pct=0.11,

            normalize=True,

            training=False,
        )
    )

    # ==========================================================
    # DATASETS
    # ==============================================================

    train_dataset = (
        NuScenesLSSDataset(

            dataroot=DATAROOT,

            version=VERSION,

            split=TRAIN_SPLIT,

            image_transform=(
                train_transform
            ),

            verbose=False,
        )
    )

    val_dataset = (
        NuScenesLSSDataset(

            dataroot=DATAROOT,

            version=VERSION,

            split=VAL_SPLIT,

            image_transform=(
                val_transform
            ),

            verbose=False,
        )
    )

    print(
        "Train samples:",
        len(
            train_dataset
        ),
    )

    print(
        "Val samples:",
        len(
            val_dataset
        ),
    )

    print(
        "Classes:",
        train_dataset.classes,
    )

    # ==========================================================
    # DATALOADERS
    # ==========================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        # ------------------------------------------------------
        # Vrai training:
        #
        # on mélange les frames à chaque epoch.
        # ------------------------------------------------------

        shuffle=True,

        num_workers=NUM_WORKERS,

        collate_fn=lss_collate_fn,

        pin_memory=(
            device.type
            == "cuda"
        ),

        drop_last=False,
    )

    val_loader = DataLoader(

        val_dataset,

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
    # ==============================================================

    target_builder = (
        CenterPointTargetBuilder(

            classes=(
                train_dataset.classes
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
    # Architecture validée précédemment:
    #
    # EfficientNet pretrained
    #
    # EfficientNet BatchNorm:
    #     frozen
    #
    # EfficientNet DropConnect:
    #     disabled
    #
    # FeatureFusion:
    #     GroupNorm
    #
    # BEV Backbone:
    #     GroupNorm
    #
    # Detection Head:
    #     GroupNorm
    #
    # ==============================================================

    model = (
        LSSDetector(

            num_classes=len(
                train_dataset.classes
            ),

            camera_pretrained=(
                CAMERA_PRETRAINED
            ),

            return_intermediates=False,
        )
    )

    model = model.to(
        device
    )

    # ==========================================================
    # LOSS
    # ==============================================================

    criterion = DetectionLoss(

        heatmap_weight=1.0,

        reg_weight=1.0,

        center_z_weight=1.0,

        dim_weight=1.0,

        rot_weight=1.0,
    )

    # ==========================================================
    # OPTIMIZER
    # ==========================================================
    #
    # AdamW:
    #
    # Adam
    # +
    # decoupled weight decay.
    #
    #
    # L'expérience précédente utilisait:
    #
    #   AdamW
    #   weight_decay = 1e-4
    #
    # On garde exactement la même configuration.
    #
    # ==============================================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,
    )

    # ==========================================================
    # DECODER
    # ==========================================================
    #
    # Seulement pour debug.
    #
    # Ce decoder n'intervient jamais dans:
    #
    #   loss
    #   backward
    #   optimizer
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
        # Threshold volontairement faible pendant
        # les expérimentations.
        # ------------------------------------------------------

        score_threshold=0.1,

        use_circle_nms=True,

        nms_min_distance=1.0,
    )

    # ==========================================================
    # MODEL INFO
    # ==============================================================

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

    print(
        "Epochs:",
        EPOCHS,
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Learning rate:",
        LEARNING_RATE,
    )

    print(
        "Weight decay:",
        WEIGHT_DECAY,
    )

    print(
        "Photometric augmentation:",
        True,
    )

    print()

    # ==========================================================
    # TRAINING LOOP
    # ==============================================================

    best_val_loss = float(
        "inf"
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        # ======================================================
        # TRAIN
        # ==============================================================

        train_losses = train_one_epoch(

            model=model,

            loader=train_loader,

            target_builder=target_builder,

            criterion=criterion,

            optimizer=optimizer,

            device=device,
        )

        # ======================================================
        # VALIDATION
        # ==============================================================

        val_losses = validate(

            model=model,

            loader=val_loader,

            target_builder=target_builder,

            criterion=criterion,

            device=device,
        )

        # ======================================================
        # LOGS
        # ==============================================================

        print(

            f"Epoch "
            f"{epoch:03d}/{EPOCHS}"
        )

        print(

            "  TRAIN | "

            f"total="
            f"{train_losses['loss']:.4f} | "

            f"hm="
            f"{train_losses['loss_heatmap']:.4f} | "

            f"reg="
            f"{train_losses['loss_reg']:.4f} | "

            f"z="
            f"{train_losses['loss_center_z']:.4f} | "

            f"dim="
            f"{train_losses['loss_dim']:.4f} | "

            f"rot="
            f"{train_losses['loss_rot']:.4f} | "

            f"GTp="
            f"{train_losses['gt_center_probability']:.4f}"
        )

        print(

            "  VAL   | "

            f"total="
            f"{val_losses['loss']:.4f} | "

            f"hm="
            f"{val_losses['loss_heatmap']:.4f} | "

            f"reg="
            f"{val_losses['loss_reg']:.4f} | "

            f"z="
            f"{val_losses['loss_center_z']:.4f} | "

            f"dim="
            f"{val_losses['loss_dim']:.4f} | "

            f"rot="
            f"{val_losses['loss_rot']:.4f} | "

            f"GTp="
            f"{val_losses['gt_center_probability']:.4f}"
        )

        # ======================================================
        # BEST VALIDATION CHECKPOINT
        # ==============================================================

        if (
            val_losses[
                "loss"
            ]
            < best_val_loss
        ):

            best_val_loss = (
                val_losses[
                    "loss"
                ]
            )

            save_checkpoint(

                path=(
                    BEST_CHECKPOINT_PATH
                ),

                epoch=epoch,

                model=model,

                optimizer=optimizer,

                train_losses=(
                    train_losses
                ),

                val_losses=(
                    val_losses
                ),

                classes=(
                    train_dataset.classes
                ),
            )

            print(

                f"  -> New best validation "
                f"loss: "
                f"{best_val_loss:.4f}"
            )

        # ======================================================
        # LAST CHECKPOINT
        # ==============================================================

        save_checkpoint(

            path=(
                LAST_CHECKPOINT_PATH
            ),

            epoch=epoch,

            model=model,

            optimizer=optimizer,

            train_losses=(
                train_losses
            ),

            val_losses=(
                val_losses
            ),

            classes=(
                train_dataset.classes
            ),
        )

        # ======================================================
        # PERIODIC VALIDATION DEBUG
        # ==============================================================

        if (
            epoch == 1
            or
            epoch % DEBUG_EVERY == 0
        ):

            print(
                "  --- validation debug ---"
            )

            inspect_validation_sample(

                model=model,

                dataset=val_dataset,

                decoder=decoder,

                device=device,

                sample_index=0,
            )

            print(
                "  ------------------------"
            )

        print()

    # ==========================================================
    # FINISHED
    # ==============================================================

    print(
        "======================================"
    )

    print(
        "TRAINING FINISHED"
    )

    print(
        "======================================"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        "Best checkpoint:",
        BEST_CHECKPOINT_PATH,
    )

    print(
        "Last checkpoint:",
        LAST_CHECKPOINT_PATH,
    )

    print(
        "======================================"
    )


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":

    main()