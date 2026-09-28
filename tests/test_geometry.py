#!/usr/bin/env python3

import math

import torch

from lss_det.models.geometry import Geometry


def identity_calibration(B=1, N=1):
    """
    Calibration identité utile pour les tests.
    """

    intrins = torch.eye(3).view(
        1, 1, 3, 3
    ).repeat(
        B, N, 1, 1
    )

    rots = torch.eye(3).view(
        1, 1, 3, 3
    ).repeat(
        B, N, 1, 1
    )

    trans = torch.zeros(
        B,
        N,
        3,
    )

    post_rots = torch.eye(3).view(
        1, 1, 3, 3
    ).repeat(
        B, N, 1, 1
    )

    post_trans = torch.zeros(
        B,
        N,
        3,
    )

    return (
        intrins,
        rots,
        trans,
        post_rots,
        post_trans,
    )


def test_frustum_shape():

    model = Geometry(
        image_size=(128, 352),
        feature_size=(8, 22),
        depth_bound=(4.0, 45.0, 1.0),
    )

    assert model.frustum.shape == (
        41,
        8,
        22,
        3,
    )


def test_frustum_depth_bounds():

    model = Geometry(
        image_size=(128, 352),
        feature_size=(8, 22),
        depth_bound=(4.0, 45.0, 1.0),
    )

    depths = model.frustum[
        :, 0, 0, 2
    ]

    expected = torch.arange(
        4.0,
        45.0,
        1.0,
    )

    assert torch.allclose(
        depths,
        expected,
    )


def test_geometry_shape():

    model = Geometry(
        image_size=(128, 352),
        feature_size=(8, 22),
        depth_bound=(4.0, 45.0, 1.0),
    )

    B = 2
    N = 4

    inputs = identity_calibration(
        B,
        N,
    )

    geometry = model(*inputs)

    assert geometry.shape == (
        2,
        4,
        41,
        8,
        22,
        3,
    )


def test_known_projection():

    """
    Test numérique contrôlé.

    Image:
        3 x 3

    Feature map:
        3 x 3

    Donc les coordonnées u,v sont:
        0,1,2

    On choisit:

        fx = 1
        fy = 1
        cx = 1
        cy = 1

    Pixel:
        u = 2
        v = 1

    Depth:
        d = 2

    Formules:

        X = (u-cx)/fx * d
          = (2-1) * 2
          = 2

        Y = (v-cy)/fy * d
          = 0

        Z = 2

    Résultat attendu:

        [2,0,2]
    """

    model = Geometry(
        image_size=(3, 3),
        feature_size=(3, 3),
        depth_bound=(1.0, 3.0, 1.0),
    )

    intrins = torch.tensor(
        [
            [
                [
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0],
                ]
            ]
        ]
    )

    rots = torch.eye(3).view(
        1, 1, 3, 3
    )

    trans = torch.zeros(
        1, 1, 3
    )

    post_rots = torch.eye(3).view(
        1, 1, 3, 3
    )

    post_trans = torch.zeros(
        1, 1, 3
    )

    geometry = model(
        intrins,
        rots,
        trans,
        post_rots,
        post_trans,
    )

    # depth bins:
    #
    # index 0 -> 1m
    # index 1 -> 2m
    #
    d_idx = 1

    # pixel:
    #
    # v = 1
    # u = 2
    #
    point = geometry[
        0,
        0,
        d_idx,
        1,
        2,
    ]

    expected = torch.tensor(
        [2.0, 0.0, 2.0]
    )

    assert torch.allclose(
        point,
        expected,
        atol=1e-6,
    )


def test_translation():

    """
    Une translation extrinsèque [1,2,3]
    doit simplement ajouter [1,2,3]
    à chaque point.
    """

    model = Geometry(
        image_size=(3, 3),
        feature_size=(3, 3),
        depth_bound=(1.0, 2.0, 1.0),
    )

    (
        intrins,
        rots,
        trans,
        post_rots,
        post_trans,
    ) = identity_calibration()

    geometry_without_translation = model(
        intrins,
        rots,
        trans,
        post_rots,
        post_trans,
    )

    trans_shifted = torch.tensor(
        [[[1.0, 2.0, 3.0]]]
    )

    geometry_with_translation = model(
        intrins,
        rots,
        trans_shifted,
        post_rots,
        post_trans,
    )

    difference = (
        geometry_with_translation
        - geometry_without_translation
    )

    expected = torch.tensor(
        [1.0, 2.0, 3.0]
    )

    assert torch.allclose(
        difference,
        expected,
        atol=1e-6,
    )


def test_rotation_90_deg():

    """
    Vérifie l'application d'une rotation
    extrinsèque de +90° autour de Z.

    [X,Y,Z]

    devient:

    [-Y,X,Z]
    """

    model = Geometry(
        image_size=(3, 3),
        feature_size=(3, 3),
        depth_bound=(1.0, 2.0, 1.0),
    )

    intrins = torch.eye(3).view(
        1, 1, 3, 3
    )

    angle = math.pi / 2.0

    rotation_z = torch.tensor(
        [
            [
                [
                    [math.cos(angle), -math.sin(angle), 0.0],
                    [math.sin(angle),  math.cos(angle), 0.0],
                    [0.0,              0.0,             1.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    trans = torch.zeros(
        1, 1, 3
    )

    post_rots = torch.eye(3).view(
        1, 1, 3, 3
    )

    post_trans = torch.zeros(
        1, 1, 3
    )

    geometry = model(
        intrins,
        rotation_z,
        trans,
        post_rots,
        post_trans,
    )

    # Avec K=I, u=2, v=0, d=1:
    #
    # Pcam = [2,0,1]
    #
    # Rz(90°):
    #
    # -> [0,2,1]
    #
    point = geometry[
        0,
        0,
        0,
        0,
        2,
    ]

    expected = torch.tensor(
        [0.0, 2.0, 1.0]
    )

    assert torch.allclose(
        point,
        expected,
        atol=1e-5,
    )


def test_post_resize_inverse():

    """
    Vérifie l'annulation d'un resize.

    Supposons:

        u_aug = 0.5 * u_original
        v_aug = 0.5 * v_original

    Donc:

        post_rots =
            diag(0.5, 0.5, 1)

    Un pixel augmenté:

        (u_aug,v_aug) = (1,1)

    doit revenir à:

        (u_original,v_original) = (2,2)
    """

    model = Geometry(
        image_size=(3, 3),
        feature_size=(3, 3),
        depth_bound=(1.0, 2.0, 1.0),
    )

    intrins = torch.eye(3).view(
        1, 1, 3, 3
    )

    rots = torch.eye(3).view(
        1, 1, 3, 3
    )

    trans = torch.zeros(
        1, 1, 3
    )

    post_rots = torch.tensor(
        [
            [
                [
                    [0.5, 0.0, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ]
        ]
    )

    post_trans = torch.zeros(
        1, 1, 3
    )

    geometry = model(
        intrins,
        rots,
        trans,
        post_rots,
        post_trans,
    )

    # Frustum cell:
    #
    # u_aug = 1
    # v_aug = 1
    # d = 1
    #
    # Undo resize:
    #
    # u_original = 2
    # v_original = 2
    #
    # K=I:
    #
    # XYZ = [2,2,1]
    #
    point = geometry[
        0,
        0,
        0,
        1,
        1,
    ]

    expected = torch.tensor(
        [2.0, 2.0, 1.0]
    )

    assert torch.allclose(
        point,
        expected,
        atol=1e-6,
    )