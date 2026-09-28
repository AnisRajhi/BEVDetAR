#!/usr/bin/env python3

import torch

from lss_det.models.voxel_pooling import VoxelPooling


def make_small_pool():
    """
    Petite grille pour les tests:

    X:
        [-1, +1[
        step = 1
        Nx = 2

    Y:
        [-1, +1[
        step = 1
        Ny = 2

    Z:
        [-1, +1[
        step = 2
        Nz = 1
    """

    return VoxelPooling(
        xbound=(-1.0, 1.0, 1.0),
        ybound=(-1.0, 1.0, 1.0),
        zbound=(-1.0, 1.0, 2.0),
    )


def test_grid_size():

    pool = make_small_pool()

    assert pool.Nx == 2
    assert pool.Ny == 2
    assert pool.Nz == 1


def test_output_shape():

    pool = make_small_pool()

    # B=2
    # N=3
    # D=2
    # H=2
    # W=2
    # C=4

    features = torch.randn(
        2,
        3,
        2,
        2,
        2,
        4,
    )

    # Tous les points à l'intérieur de la grille.
    geometry = torch.zeros(
        2,
        3,
        2,
        2,
        2,
        3,
    )

    bev = pool(
        features,
        geometry,
    )

    assert bev.shape == (
        2,
        4,
        2,
        2,
    )


def test_single_point_known_voxel():

    pool = make_small_pool()

    # Un seul point.
    #
    # Shape:
    #
    # [B,N,D,H,W,C]
    #
    features = torch.tensor(
        [[[[[[
            2.0,
            3.0,
        ]]]]]]
    )

    # XYZ:
    #
    # X = -0.2
    #
    # ix =
    # floor((-0.2 - (-1)) / 1)
    # = floor(0.8)
    # = 0
    #
    # Y = +0.4
    #
    # iy =
    # floor((0.4 - (-1)) / 1)
    # = floor(1.4)
    # = 1
    #
    # Z = 0
    #
    # iz = 0

    geometry = torch.tensor(
        [[[[[[
            -0.2,
             0.4,
             0.0,
        ]]]]]]
    )

    bev = pool(
        features,
        geometry,
    )

    assert bev.shape == (
        1,
        2,
        2,
        2,
    )

    # La feature doit être uniquement
    # dans voxel:
    #
    # ix=0
    # iy=1

    expected = torch.tensor(
        [2.0, 3.0]
    )

    actual = bev[
        0,
        :,
        0,
        1,
    ]

    assert torch.allclose(
        actual,
        expected,
    )

    # La somme totale du BEV doit être:
    #
    # 2 + 3 = 5

    assert torch.allclose(
        bev.sum(),
        torch.tensor(5.0),
    )


def test_same_voxel_features_are_summed():

    pool = make_small_pool()

    # Deux points.
    #
    # On utilise ici D=2:
    #
    # [B=1,N=1,D=2,H=1,W=1,C=2]

    features = torch.tensor(
        [
            [
                [
                    [[[1.0, 2.0]]],
                    [[[3.0, 4.0]]],
                ]
            ]
        ]
    )

    assert features.shape == (
        1, 1, 2, 1, 1, 2
    )

    # Les deux XYZ tombent dans le même voxel:
    #
    # ix=0
    # iy=1
    # iz=0

    geometry = torch.tensor(
        [
            [
                [
                    [[[-0.20, 0.20, 0.0]]],
                    [[[-0.10, 0.30, 0.0]]],
                ]
            ]
        ]
    )

    bev = pool(
        features,
        geometry,
    )

    # Somme attendue:
    #
    # [1,2] + [3,4]
    #
    # =
    #
    # [4,6]

    expected = torch.tensor(
        [4.0, 6.0]
    )

    actual = bev[
        0,
        :,
        0,
        1,
    ]

    assert torch.allclose(
        actual,
        expected,
    )


def test_different_voxels_are_not_mixed():

    pool = make_small_pool()

    features = torch.tensor(
        [
            [
                [
                    [[[1.0]]],
                    [[[5.0]]],
                ]
            ]
        ]
    )

    # Premier point:
    #
    # (-0.5,-0.5)
    #
    # -> voxel (0,0)
    #
    # Deuxième:
    #
    # (+0.5,+0.5)
    #
    # -> voxel (1,1)

    geometry = torch.tensor(
        [
            [
                [
                    [[[-0.5, -0.5, 0.0]]],
                    [[[ 0.5,  0.5, 0.0]]],
                ]
            ]
        ]
    )

    bev = pool(
        features,
        geometry,
    )

    assert torch.allclose(
        bev[0, 0, 0, 0],
        torch.tensor(1.0),
    )

    assert torch.allclose(
        bev[0, 0, 1, 1],
        torch.tensor(5.0),
    )


def test_out_of_bounds_point_is_removed():

    pool = make_small_pool()

    features = torch.tensor(
        [[[[[[10.0]]]]]]
    )

    # X = 5m:
    #
    # totalement hors de:
    #
    # [-1,+1[
    #
    geometry = torch.tensor(
        [[[[[[5.0, 0.0, 0.0]]]]]]
    )

    bev = pool(
        features,
        geometry,
    )

    # Aucun point conservé.
    assert torch.allclose(
        bev.sum(),
        torch.tensor(0.0),
    )


def test_batches_are_not_mixed():

    pool = make_small_pool()

    # B=2
    #
    # Les deux points ont EXACTEMENT
    # le même XYZ mais appartiennent
    # à deux samples différents.

    features = torch.tensor(
        [
            [[[[[2.0]]]]],
            [[[[[7.0]]]]],
        ]
    )

    assert features.shape == (
        2, 1, 1, 1, 1, 1
    )

    geometry = torch.tensor(
        [
            [[[[[0.0, 0.0, 0.0]]]]],
            [[[[[0.0, 0.0, 0.0]]]]],
        ]
    )

    bev = pool(
        features,
        geometry,
    )

    # XYZ=(0,0,0)
    #
    # donne:
    #
    # ix=1
    # iy=1

    assert torch.allclose(
        bev[0, 0, 1, 1],
        torch.tensor(2.0),
    )

    assert torch.allclose(
        bev[1, 0, 1, 1],
        torch.tensor(7.0),
    )


def test_gradient_reaches_features():

    pool = make_small_pool()

    features = torch.tensor(
        [[[[[[2.0, 3.0]]]]]],
        requires_grad=True,
    )

    geometry = torch.tensor(
        [[[[[[0.0, 0.0, 0.0]]]]]]
    )

    bev = pool(
        features,
        geometry,
    )

    loss = bev.sum()

    loss.backward()

    # Le gradient doit revenir vers
    # les lifted features.
    assert features.grad is not None

    assert torch.allclose(
        features.grad,
        torch.ones_like(features),
    )