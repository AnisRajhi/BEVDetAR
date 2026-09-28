#!/usr/bin/env python3

import torch

from lss_det.models.lss_detector import (
    LSSDetector,
)


def make_calibration(
    B: int,
    N: int,
):

    # ==========================================================
    # Intrinsics artificielles plausibles
    # ==========================================================
    #
    # image:
    #
    # H = 128
    # W = 352
    #
    # centre approximatif:
    #
    # cx = 176
    # cy = 64
    #

    K = torch.tensor(
        [
            [200.0,   0.0, 176.0],
            [  0.0, 200.0,  64.0],
            [  0.0,   0.0,   1.0],
        ],
        dtype=torch.float32,
    )

    intrins = (
        K
        .view(
            1,
            1,
            3,
            3,
        )
        .repeat(
            B,
            N,
            1,
            1,
        )
    )

    # ==========================================================
    # Camera optical frame -> ego frame
    # ==========================================================
    #
    # Optical:
    #
    # X = right
    # Y = down
    # Z = forward
    #
    # Ego:
    #
    # X = forward
    # Y = left
    # Z = up
    #
    #
    # Donc:
    #
    # Xego = Zcam
    # Yego = -Xcam
    # Zego = -Ycam
    #

    R = torch.tensor(
        [
            [0.0,  0.0,  1.0],
            [-1.0, 0.0,  0.0],
            [0.0, -1.0,  0.0],
        ],
        dtype=torch.float32,
    )

    rots = (
        R
        .view(
            1,
            1,
            3,
            3,
        )
        .repeat(
            B,
            N,
            1,
            1,
        )
    )

    # Caméra à 1.2 m de hauteur.

    trans = torch.zeros(
        B,
        N,
        3,
        dtype=torch.float32,
    )

    trans[
        ...,
        2,
    ] = 1.2

    # ==========================================================
    # Pas d'augmentation image
    # ==========================================================

    post_rots = (
        torch.eye(
            3,
            dtype=torch.float32,
        )
        .view(
            1,
            1,
            3,
            3,
        )
        .repeat(
            B,
            N,
            1,
            1,
        )
    )

    post_trans = torch.zeros(
        B,
        N,
        3,
        dtype=torch.float32,
    )

    return (
        intrins,
        rots,
        trans,
        post_rots,
        post_trans,
    )


def test_complete_forward():

    B = 1
    N = 2
    K = 6

    # IMPORTANT:
    #
    # pretrained=False
    #
    # pour qu'un test unitaire ne tente
    # jamais de télécharger des poids.

    model = LSSDetector(
        num_classes=K,
        camera_pretrained=False,
        return_intermediates=True,
    )

    model.eval()

    images = torch.randn(
        B,
        N,
        3,
        128,
        352,
    )

    calibration = make_calibration(
        B,
        N,
    )

    with torch.no_grad():

        outputs = model(
            images,
            *calibration,
        )

    # ==========================================================
    # CameraEncoder
    # ==========================================================

    assert (
        outputs[
            "camera_features"
        ].shape
        ==
        (
            B,
            N,
            512,
            8,
            22,
        )
    )

    # ==========================================================
    # Lift
    # ==========================================================

    assert (
        outputs[
            "depth_probs"
        ].shape
        ==
        (
            B,
            N,
            41,
            8,
            22,
        )
    )

    assert (
        outputs[
            "lifted_features"
        ].shape
        ==
        (
            B,
            N,
            41,
            8,
            22,
            64,
        )
    )

    # ==========================================================
    # Geometry
    # ==========================================================

    assert (
        outputs[
            "geometry"
        ].shape
        ==
        (
            B,
            N,
            41,
            8,
            22,
            3,
        )
    )

    # ==========================================================
    # SPLAT
    # ==========================================================

    assert (
        outputs[
            "bev_raw"
        ].shape
        ==
        (
            B,
            64,
            200,
            200,
        )
    )

    # ==========================================================
    # BEV Backbone
    # ==========================================================

    assert (
        outputs[
            "bev_features"
        ].shape
        ==
        (
            B,
            128,
            200,
            200,
        )
    )

    # ==========================================================
    # Detection Head
    # ==========================================================

    predictions = outputs[
        "predictions"
    ]

    assert (
        predictions[
            "heatmap"
        ].shape
        ==
        (
            B,
            K,
            200,
            200,
        )
    )

    assert (
        predictions[
            "reg"
        ].shape
        ==
        (
            B,
            2,
            200,
            200,
        )
    )

    assert (
        predictions[
            "center_z"
        ].shape
        ==
        (
            B,
            1,
            200,
            200,
        )
    )

    assert (
        predictions[
            "dim"
        ].shape
        ==
        (
            B,
            3,
            200,
            200,
        )
    )

    assert (
        predictions[
            "rot"
        ].shape
        ==
        (
            B,
            2,
            200,
            200,
        )
    )


def test_depth_probabilities():

    B = 1
    N = 2

    model = LSSDetector(
        num_classes=3,
        camera_pretrained=False,
        return_intermediates=True,
    )

    model.eval()

    images = torch.randn(
        B,
        N,
        3,
        128,
        352,
    )

    calibration = make_calibration(
        B,
        N,
    )

    with torch.no_grad():

        outputs = model(
            images,
            *calibration,
        )

    depth_probs = outputs[
        "depth_probs"
    ]

    sums = depth_probs.sum(
        dim=2
    )

    assert torch.allclose(
        sums,
        torch.ones_like(
            sums
        ),
        atol=1e-5,
    )


def test_bev_is_finite():

    model = LSSDetector(
        num_classes=3,
        camera_pretrained=False,
        return_intermediates=True,
    )

    model.eval()

    B = 1
    N = 2

    images = torch.randn(
        B,
        N,
        3,
        128,
        352,
    )

    calibration = make_calibration(
        B,
        N,
    )

    with torch.no_grad():

        outputs = model(
            images,
            *calibration,
        )

    assert torch.isfinite(
        outputs["bev_raw"]
    ).all()

    assert torch.isfinite(
        outputs["bev_features"]
    ).all()


def test_variable_number_of_cameras():

    # ==========================================================
    # nuScenes:
    # N=6
    #
    # Robot:
    # N=4
    #
    # Le réseau doit supporter les deux.
    # ==========================================================

    model = LSSDetector(
        num_classes=3,
        camera_pretrained=False,
        return_intermediates=False,
    )

    model.eval()

    for N in [
        2,
        4,
        6,
    ]:

        B = 1

        images = torch.randn(
            B,
            N,
            3,
            128,
            352,
        )

        calibration = make_calibration(
            B,
            N,
        )

        with torch.no_grad():

            predictions = model(
                images,
                *calibration,
            )

        assert (
            predictions[
                "heatmap"
            ].shape
            ==
            (
                1,
                3,
                200,
                200,
            )
        )
