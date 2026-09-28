#!/usr/bin/env python3

import torch

from lss_det.losses.detection_loss import (
    DetectionLoss,
    gather_feature_map,
    gaussian_focal_loss,
    masked_l1_loss,
)


def test_gather_feature_map():

    # ----------------------------------------------------------
    # Feature map:
    #
    # B=1
    # C=1
    # Nx=2
    # Ny=3
    #
    # Spatial:
    #
    # [ [0,1,2],
    #   [3,4,5] ]
    # ----------------------------------------------------------

    prediction = torch.tensor(
        [[[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
        ]]]
    )

    # ix=1, iy=1
    #
    # index = 1*3 + 1 = 4

    indices = torch.tensor(
        [[4]]
    )

    gathered = gather_feature_map(
        prediction,
        indices,
    )

    assert gathered.shape == (
        1,
        1,
        1,
    )

    assert torch.allclose(
        gathered[0, 0, 0],
        torch.tensor(4.0),
    )


def test_masked_l1_loss_zero():

    prediction = torch.tensor(
        [[[
            1.0,
            2.0,
        ]]]
    )

    target = prediction.clone()

    mask = torch.tensor(
        [[True]]
    )

    loss = masked_l1_loss(
        prediction,
        target,
        mask,
    )

    assert torch.allclose(
        loss,
        torch.tensor(0.0),
    )


def test_mask_ignores_invalid_objects():

    prediction = torch.tensor(
        [[
            [1.0, 2.0],
            [100.0, 100.0],
        ]]
    )

    target = torch.tensor(
        [[
            [1.0, 2.0],
            [0.0, 0.0],
        ]]
    )

    mask = torch.tensor(
        [[
            True,
            False,
        ]]
    )

    loss = masked_l1_loss(
        prediction,
        target,
        mask,
    )

    # Le deuxième objet est ignoré.
    assert torch.allclose(
        loss,
        torch.tensor(0.0),
    )


def test_good_heatmap_has_lower_loss():

    target = torch.zeros(
        1,
        1,
        5,
        5,
    )

    target[
        0,
        0,
        2,
        2,
    ] = 1.0

    # Mauvaise prédiction:
    #
    # logits 0 => sigmoid = 0.5 partout

    bad_logits = torch.zeros_like(
        target
    )

    # Meilleure prédiction:
    #
    # centre fortement positif
    # background fortement négatif

    good_logits = torch.full_like(
        target,
        -5.0,
    )

    good_logits[
        0,
        0,
        2,
        2,
    ] = 5.0

    bad_loss = gaussian_focal_loss(
        bad_logits,
        target,
    )

    good_loss = gaussian_focal_loss(
        good_logits,
        target,
    )

    assert good_loss < bad_loss


def test_complete_detection_loss():

    B = 1
    K = 2
    Nx = 10
    Ny = 10

    predictions = {
        "heatmap": torch.randn(
            B,
            K,
            Nx,
            Ny,
            requires_grad=True,
        ),

        "reg": torch.randn(
            B,
            2,
            Nx,
            Ny,
            requires_grad=True,
        ),

        "center_z": torch.randn(
            B,
            1,
            Nx,
            Ny,
            requires_grad=True,
        ),

        "dim": torch.randn(
            B,
            3,
            Nx,
            Ny,
            requires_grad=True,
        ),

        "rot": torch.randn(
            B,
            2,
            Nx,
            Ny,
            requires_grad=True,
        ),
    }

    # Un objet à:
    #
    # ix = 4
    # iy = 5
    #
    center_index = (
        4 * Ny
        + 5
    )

    max_objects = 5

    heatmap = torch.zeros(
        B,
        K,
        Nx,
        Ny,
    )

    heatmap[
        0,
        0,
        4,
        5,
    ] = 1.0

    targets = {
        "heatmap": heatmap,

        "indices": torch.tensor(
            [[
                center_index,
                0,
                0,
                0,
                0,
            ]]
        ),

        "mask": torch.tensor(
            [[
                True,
                False,
                False,
                False,
                False,
            ]]
        ),

        "offset": torch.zeros(
            B,
            max_objects,
            2,
        ),

        "center_z": torch.zeros(
            B,
            max_objects,
            1,
        ),

        "dim": torch.ones(
            B,
            max_objects,
            3,
        ),

        "rot": torch.zeros(
            B,
            max_objects,
            2,
        ),
    }

    # yaw = 0
    #
    # sin = 0
    # cos = 1

    targets["rot"][
        0,
        0,
        1,
    ] = 1.0

    criterion = DetectionLoss()

    losses = criterion(
        predictions,
        targets,
    )

    assert "loss" in losses

    assert torch.isfinite(
        losses["loss"]
    )

    assert losses["loss"] > 0

    losses["loss"].backward()

    # Vérifier gradient sur les heads.

    assert (
        predictions["heatmap"].grad
        is not None
    )

    assert (
        predictions["reg"].grad
        is not None
    )

    assert (
        predictions["dim"].grad
        is not None
    )


def test_perfect_regression_has_zero_regression_loss():

    B = 1
    M = 1

    pred = torch.tensor(
        [[[
            0.3,
            0.7,
        ]]]
    )

    target = pred.clone()

    mask = torch.tensor(
        [[True]]
    )

    loss = masked_l1_loss(
        pred,
        target,
        mask,
    )

    assert torch.allclose(
        loss,
        torch.tensor(0.0),
    )
