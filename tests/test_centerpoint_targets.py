#!/usr/bin/env python3

import math

import torch

from lss_det.targets.centerpoint_targets import (
    CenterPointTargetBuilder,
)


def make_builder():

    return CenterPointTargetBuilder(
        classes=[
            "car",
            "pedestrian",
        ],
        xbound=(-50.0, 50.0, 0.5),
        ybound=(-50.0, 50.0, 0.5),
        max_objects=20,
        min_radius=1,
    )


def test_target_shapes():

    builder = make_builder()

    boxes = [
        torch.tensor(
            [
                [
                    0.0,   # x
                    0.0,   # y
                    1.0,   # z
                    4.0,   # l
                    2.0,   # w
                    1.5,   # h
                    0.0,   # yaw
                ]
            ],
            dtype=torch.float32,
        )
    ]

    labels = [
        torch.tensor(
            [0],
            dtype=torch.long,
        )
    ]

    targets = builder(
        boxes,
        labels,
    )

    assert targets["heatmap"].shape == (
        1,
        2,
        200,
        200,
    )

    assert targets["indices"].shape == (
        1,
        20,
    )

    assert targets["offset"].shape == (
        1,
        20,
        2,
    )

    assert targets["center_z"].shape == (
        1,
        20,
        1,
    )

    assert targets["dim"].shape == (
        1,
        20,
        3,
    )

    assert targets["rot"].shape == (
        1,
        20,
        2,
    )


def test_known_center():

    builder = make_builder()

    # x = 12.3
    # y = -3.7
    #
    # gx = 124.6
    # gy = 92.6

    boxes = [
        torch.tensor(
            [[
                12.3,
                -3.7,
                0.8,
                4.2,
                1.8,
                1.5,
                0.0,
            ]]
        )
    ]

    labels = [
        torch.tensor([0])
    ]

    targets = builder(
        boxes,
        labels,
    )

    ix = 124
    iy = 92

    # Peak central de la Gaussian.

    assert torch.allclose(
        targets["heatmap"][
            0,
            0,
            ix,
            iy,
        ],
        torch.tensor(1.0),
    )

    # Linear index.

    expected_index = (
        ix * 200
        + iy
    )

    assert (
        targets["indices"][0, 0].item()
        ==
        expected_index
    )


def test_known_offset():

    builder = make_builder()

    boxes = [
        torch.tensor(
            [[
                12.3,
                -3.7,
                0.8,
                4.2,
                1.8,
                1.5,
                0.0,
            ]]
        )
    ]

    labels = [
        torch.tensor([0])
    ]

    targets = builder(
        boxes,
        labels,
    )

    expected = torch.tensor(
        [0.6, 0.6]
    )

    assert torch.allclose(
        targets["offset"][
            0,
            0,
        ],
        expected,
        atol=1e-5,
    )


def test_box_attributes():

    builder = make_builder()

    yaw = 0.3

    boxes = [
        torch.tensor(
            [[
                1.0,
                2.0,
                0.8,
                4.2,
                1.8,
                1.5,
                yaw,
            ]]
        )
    ]

    labels = [
        torch.tensor([0])
    ]

    targets = builder(
        boxes,
        labels,
    )

    assert torch.allclose(
        targets["center_z"][0, 0],
        torch.tensor([0.8]),
    )

    assert torch.allclose(
        targets["dim"][0, 0],
        torch.tensor(
            [4.2, 1.8, 1.5]
        ),
    )

    expected_rot = torch.tensor(
        [
            math.sin(yaw),
            math.cos(yaw),
        ]
    )

    assert torch.allclose(
        targets["rot"][0, 0],
        expected_rot,
        atol=1e-6,
    )


def test_different_classes():

    builder = make_builder()

    boxes = [
        torch.tensor(
            [
                [
                    0.0,
                    0.0,
                    1.0,
                    4.0,
                    2.0,
                    1.5,
                    0.0,
                ],
                [
                    10.0,
                    10.0,
                    1.0,
                    0.8,
                    0.8,
                    1.8,
                    0.0,
                ],
            ]
        )
    ]

    labels = [
        torch.tensor(
            [
                0,  # car
                1,  # pedestrian
            ]
        )
    ]

    targets = builder(
        boxes,
        labels,
    )

    car_ix = 100
    car_iy = 100

    ped_ix = 120
    ped_iy = 120

    assert (
        targets["heatmap"][
            0,
            0,
            car_ix,
            car_iy,
        ]
        == 1
    )

    assert (
        targets["heatmap"][
            0,
            1,
            ped_ix,
            ped_iy,
        ]
        == 1
    )


def test_outside_box_is_ignored():

    builder = make_builder()

    boxes = [
        torch.tensor(
            [[
                100.0,  # outside X
                0.0,
                1.0,
                4.0,
                2.0,
                1.5,
                0.0,
            ]]
        )
    ]

    labels = [
        torch.tensor([0])
    ]

    targets = builder(
        boxes,
        labels,
    )

    assert (
        targets["heatmap"].sum()
        == 0
    )

    assert (
        targets["mask"].sum()
        == 0
    )


def test_mask():

    builder = make_builder()

    boxes = [
        torch.tensor(
            [
                [
                    0.0,
                    0.0,
                    1.0,
                    4.0,
                    2.0,
                    1.5,
                    0.0,
                ],
                [
                    5.0,
                    5.0,
                    1.0,
                    4.0,
                    2.0,
                    1.5,
                    0.0,
                ],
            ]
        )
    ]

    labels = [
        torch.tensor(
            [0, 0]
        )
    ]

    targets = builder(
        boxes,
        labels,
    )

    assert (
        targets["mask"].sum().item()
        == 2
    )
