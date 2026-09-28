#!/usr/bin/env python3

import math

import torch

from lss_det.decoding.decode_boxes import (
    CenterPointDecoder,
    local_maximum_suppression,
)


def make_predictions(
    B=1,
    K=1,
    Nx=4,
    Ny=5,
):

    return {
        "heatmap": torch.full(
            (
                B,
                K,
                Nx,
                Ny,
            ),
            -10.0,
        ),

        "reg": torch.zeros(
            B,
            2,
            Nx,
            Ny,
        ),

        "center_z": torch.zeros(
            B,
            1,
            Nx,
            Ny,
        ),

        "dim": torch.ones(
            B,
            3,
            Nx,
            Ny,
        ),

        "rot": torch.zeros(
            B,
            2,
            Nx,
            Ny,
        ),
    }


def test_known_box_decode():

    decoder = CenterPointDecoder(
        xbound=(0.0, 4.0, 1.0),
        ybound=(0.0, 5.0, 1.0),
        top_k=10,
        score_threshold=0.5,
        use_circle_nms=False,
    )

    predictions = make_predictions()

    ix = 2
    iy = 3

    # Centre objet très probable.
    predictions[
        "heatmap"
    ][
        0,
        0,
        ix,
        iy,
    ] = 10.0

    # Offset.
    predictions[
        "reg"
    ][
        0,
        :,
        ix,
        iy,
    ] = torch.tensor(
        [0.25, 0.50]
    )

    # Z.
    predictions[
        "center_z"
    ][
        0,
        0,
        ix,
        iy,
    ] = 0.8

    # Dimensions.
    predictions[
        "dim"
    ][
        0,
        :,
        ix,
        iy,
    ] = torch.tensor(
        [4.2, 1.8, 1.5]
    )

    # yaw = 0
    #
    # sin = 0
    # cos = 1
    predictions[
        "rot"
    ][
        0,
        :,
        ix,
        iy,
    ] = torch.tensor(
        [0.0, 1.0]
    )

    output = decoder(
        predictions
    )[0]

    assert output["boxes"].shape == (
        1,
        7,
    )

    box = output[
        "boxes"
    ][0]

    # x =
    #
    # 0 + (2 + .25) * 1
    #
    # = 2.25
    #
    expected = torch.tensor(
        [
            2.25,
            3.50,
            0.80,
            4.20,
            1.80,
            1.50,
            0.0,
        ]
    )

    assert torch.allclose(
        box,
        expected,
        atol=1e-4,
    )


def test_yaw_decode():

    decoder = CenterPointDecoder(
        xbound=(0.0, 4.0, 1.0),
        ybound=(0.0, 5.0, 1.0),
        score_threshold=0.5,
        use_circle_nms=False,
    )

    predictions = make_predictions()

    predictions[
        "heatmap"
    ][
        0,
        0,
        1,
        1,
    ] = 10.0

    angle = math.pi / 2.0

    predictions[
        "rot"
    ][
        0,
        :,
        1,
        1,
    ] = torch.tensor(
        [
            math.sin(angle),
            math.cos(angle),
        ]
    )

    output = decoder(
        predictions
    )[0]

    yaw = output[
        "boxes"
    ][
        0,
        6,
    ]

    assert torch.allclose(
        yaw,
        torch.tensor(angle),
        atol=1e-5,
    )


def test_class_decode():

    decoder = CenterPointDecoder(
        xbound=(0.0, 4.0, 1.0),
        ybound=(0.0, 5.0, 1.0),
        top_k=5,
        score_threshold=0.5,
        use_circle_nms=False,
    )

    predictions = make_predictions(
        K=3
    )

    predictions[
        "heatmap"
    ][
        0,
        2,
        1,
        1,
    ] = 10.0

    output = decoder(
        predictions
    )[0]

    assert (
        output["labels"][0].item()
        == 2
    )


def test_low_score_removed():

    decoder = CenterPointDecoder(
        xbound=(0.0, 4.0, 1.0),
        ybound=(0.0, 5.0, 1.0),
        score_threshold=0.9,
        use_circle_nms=False,
    )

    predictions = make_predictions()

    # sigmoid(0) = 0.5
    predictions[
        "heatmap"
    ][
        0,
        0,
        1,
        1,
    ] = 0.0

    output = decoder(
        predictions
    )[0]

    assert (
        output["boxes"].shape[0]
        == 0
    )


def test_local_maximum_suppression():

    heatmap = torch.tensor(
        [[[
            [0.1, 0.2, 0.1],
            [0.3, 0.9, 0.8],
            [0.1, 0.7, 0.2],
        ]]]
    )

    output = local_maximum_suppression(
        heatmap,
        kernel_size=3,
    )

    assert torch.allclose(
        output[
            0,
            0,
            1,
            1,
        ],
        torch.tensor(0.9),
    )

    assert (
        output.sum()
        == torch.tensor(0.9)
    )


def test_circle_nms_removes_duplicate():

    decoder = CenterPointDecoder(
        xbound=(0.0, 10.0, 1.0),
        ybound=(0.0, 10.0, 1.0),
        top_k=10,
        score_threshold=0.5,
        use_circle_nms=True,
        nms_min_distance=1.5,
    )

    predictions = make_predictions(
        Nx=10,
        Ny=10,
    )

    # Deux centres voisins de même classe.

    predictions[
        "heatmap"
    ][
        0,
        0,
        4,
        4,
    ] = 10.0

    predictions[
        "heatmap"
    ][
        0,
        0,
        5,
        4,
    ] = 9.0

    output = decoder(
        predictions
    )[0]

    # Distance = 1m,
    # NMS threshold = 1.5m
    #
    # Donc un seul objet conservé.

    assert (
        output["boxes"].shape[0]
        == 1
    )


def test_batches_are_separate():

    decoder = CenterPointDecoder(
        xbound=(0.0, 4.0, 1.0),
        ybound=(0.0, 5.0, 1.0),
        score_threshold=0.5,
        use_circle_nms=False,
    )

    predictions = make_predictions(
        B=2
    )

    predictions[
        "heatmap"
    ][
        0,
        0,
        1,
        1,
    ] = 10.0

    predictions[
        "heatmap"
    ][
        1,
        0,
        2,
        2,
    ] = 10.0

    output = decoder(
        predictions
    )

    assert len(output) == 2

    assert (
        output[0]["boxes"].shape[0]
        == 1
    )

    assert (
        output[1]["boxes"].shape[0]
        == 1
    )
