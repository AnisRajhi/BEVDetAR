#!/usr/bin/env python3

import torch

from lss_det.models.camera_encoder import CameraEncoder


def test_camera_encoder_shape():

    model = CameraEncoder(
        pretrained=False
    )

    model.eval()

    B = 2
    N = 4

    images = torch.randn(
        B,
        N,
        3,
        128,
        352,
    )

    with torch.no_grad():
        output = model(images)

    assert output.shape == (
        B,
        N,
        512,
        8,
        22,
    )


def test_camera_encoder_different_camera_count():

    model = CameraEncoder(
        pretrained=False
    )

    model.eval()

    # Simulation nuScenes:
    images = torch.randn(
        1,
        6,
        3,
        128,
        352,
    )

    with torch.no_grad():
        output = model(images)

    assert output.shape == (
        1,
        6,
        512,
        8,
        22,
    )


def test_camera_encoder_backward():

    model = CameraEncoder(
        pretrained=False
    )

    model.train()

    images = torch.randn(
        1,
        2,
        3,
        128,
        352,
    )

    output = model(images)

    # Loss artificielle uniquement pour vérifier
    # que le gradient traverse le CameraEncoder.
    loss = output.mean()

    loss.backward()

    # Vérifier qu'au moins un paramètre
    # a reçu un gradient.
    has_gradient = any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert has_gradient