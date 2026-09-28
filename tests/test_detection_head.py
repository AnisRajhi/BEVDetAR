#!/usr/bin/env python3

import torch

from lss_det.models.detection_head import (
    DetectionHead,
)


def make_head():

    return DetectionHead(
        in_channels=128,
        shared_channels=128,
        head_channels=64,
        num_classes=6,
    )


def test_detection_head_shapes():

    model = make_head()

    model.eval()

    x = torch.randn(
        2,
        128,
        200,
        200,
    )

    with torch.no_grad():

        output = model(x)

    assert output["heatmap"].shape == (
        2,
        6,
        200,
        200,
    )

    assert output["reg"].shape == (
        2,
        2,
        200,
        200,
    )

    assert output["center_z"].shape == (
        2,
        1,
        200,
        200,
    )

    assert output["dim"].shape == (
        2,
        3,
        200,
        200,
    )

    assert output["rot"].shape == (
        2,
        2,
        200,
        200,
    )


def test_different_class_count():

    model = DetectionHead(
        num_classes=3
    )

    model.eval()

    x = torch.randn(
        1,
        128,
        100,
        100,
    )

    with torch.no_grad():

        output = model(x)

    assert output["heatmap"].shape == (
        1,
        3,
        100,
        100,
    )


def test_spatial_resolution_is_preserved():

    model = make_head()

    model.eval()

    x = torch.randn(
        1,
        128,
        160,
        240,
    )

    with torch.no_grad():

        output = model(x)

    for tensor in output.values():

        assert tensor.shape[-2:] == (
            160,
            240,
        )


def test_heatmap_initial_probability_is_low():

    model = make_head()

    model.eval()

    # Input nul pour vérifier principalement
    # l'effet du bias final.

    x = torch.zeros(
        1,
        128,
        20,
        20,
    )

    with torch.no_grad():

        logits = model(x)["heatmap"]

        probabilities = torch.sigmoid(
            logits
        )

    # On ne demande pas exactement 0.1,
    # puisque les couches précédentes existent.
    #
    # Mais au démarrage on veut clairement
    # être bien sous 0.5 en moyenne.

    assert probabilities.mean() < 0.2


def test_detection_head_backward():

    model = make_head()

    model.train()

    x = torch.randn(
        1,
        128,
        50,
        50,
        requires_grad=True,
    )

    output = model(x)

    # Loss artificielle simplement pour
    # tester le gradient.

    loss = (
        output["heatmap"].mean()
        + output["reg"].mean()
        + output["center_z"].mean()
        + output["dim"].mean()
        + output["rot"].mean()
    )

    loss.backward()

    assert x.grad is not None

    # Toutes les branches doivent avoir
    # reçu des gradients.

    assert (
        model.heatmap_head.net[-1]
        .weight.grad
        is not None
    )

    assert (
        model.reg_head.net[-1]
        .weight.grad
        is not None
    )

    assert (
        model.center_z_head.net[-1]
        .weight.grad
        is not None
    )

    assert (
        model.dim_head.net[-1]
        .weight.grad
        is not None
    )

    assert (
        model.rot_head.net[-1]
        .weight.grad
        is not None
    )


def test_wrong_input_channels():

    model = make_head()

    x = torch.randn(
        1,
        64,
        200,
        200,
    )

    try:

        model(x)

        assert False

    except ValueError:

        pass