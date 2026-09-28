#!/usr/bin/env python3

import torch

from lss_det.models.bev_backbone import (
    BEVBackbone,
)


def test_bev_backbone_shape():

    model = BEVBackbone(
        in_channels=64,
        out_channels=128,
    )

    model.eval()

    x = torch.randn(
        2,
        64,
        200,
        200,
    )

    with torch.no_grad():

        y = model(x)

    assert y.shape == (
        2,
        128,
        200,
        200,
    )


def test_bev_backbone_batch_size_one():

    model = BEVBackbone()

    model.eval()

    x = torch.randn(
        1,
        64,
        200,
        200,
    )

    with torch.no_grad():

        y = model(x)

    assert y.shape == (
        1,
        128,
        200,
        200,
    )


def test_bev_backbone_different_spatial_size():

    """
    Vérifie qu'on n'est pas artificiellement
    bloqué à 200x200.

    Grâce aux F.interpolate(size=...),
    le réseau doit revenir à la taille
    d'entrée.
    """

    model = BEVBackbone()

    model.eval()

    x = torch.randn(
        1,
        64,
        160,
        240,
    )

    with torch.no_grad():

        y = model(x)

    assert y.shape == (
        1,
        128,
        160,
        240,
    )


def test_bev_backbone_backward():

    model = BEVBackbone()

    model.train()

    x = torch.randn(
        1,
        64,
        200,
        200,
        requires_grad=True,
    )

    y = model(x)

    loss = y.mean()

    loss.backward()

    assert x.grad is not None

    # Vérifier que les paramètres
    # du réseau reçoivent également
    # des gradients.

    has_gradient = any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert has_gradient


def test_wrong_channel_count():

    model = BEVBackbone(
        in_channels=64,
    )

    x = torch.randn(
        1,
        32,
        200,
        200,
    )

    try:

        model(x)

        assert False, (
            "Expected ValueError"
        )

    except ValueError:

        pass
