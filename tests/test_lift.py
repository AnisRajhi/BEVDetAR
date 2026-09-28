#!/usr/bin/env python3

import torch

from lss_det.models.lift import Lift


def test_lift_shapes():

    model = Lift(
        in_channels=512,
        num_depth_bins=41,
        context_channels=64,
    )

    model.eval()

    B = 2
    N = 4

    x = torch.randn(
        B,
        N,
        512,
        8,
        22,
    )

    with torch.no_grad():
        outputs = model(x)

    assert outputs["depth_logits"].shape == (
        B,
        N,
        41,
        8,
        22,
    )

    assert outputs["depth_probs"].shape == (
        B,
        N,
        41,
        8,
        22,
    )

    assert outputs["context"].shape == (
        B,
        N,
        64,
        8,
        22,
    )

    assert outputs["lifted_features"].shape == (
        B,
        N,
        41,
        8,
        22,
        64,
    )


def test_depth_probabilities_sum_to_one():

    model = Lift()

    model.eval()

    x = torch.randn(
        1,
        4,
        512,
        8,
        22,
    )

    with torch.no_grad():
        outputs = model(x)

    depth_probs = outputs["depth_probs"]

    # Shape:
    #
    # [B,N,D,H,W]
    #
    # Donc on somme suivant D = dim 2.
    #
    depth_sum = depth_probs.sum(dim=2)

    expected = torch.ones_like(depth_sum)

    assert torch.allclose(
        depth_sum,
        expected,
        atol=1e-5,
    )


def test_depth_probabilities_are_valid():

    model = Lift()

    model.eval()

    x = torch.randn(
        1,
        2,
        512,
        8,
        22,
    )

    with torch.no_grad():
        outputs = model(x)

    depth_probs = outputs["depth_probs"]

    # Les probabilités doivent être >= 0.
    assert torch.all(depth_probs >= 0.0)

    # Et <= 1.
    assert torch.all(depth_probs <= 1.0)


def test_lifted_feature_computation():

    """
    Vérifie explicitement que:

        lifted[d,v,u,c]
        =
        depth[d,v,u] * context[c,v,u]
    """

    model = Lift()

    model.eval()

    x = torch.randn(
        1,
        1,
        512,
        8,
        22,
    )

    with torch.no_grad():
        outputs = model(x)

    depth = outputs["depth_probs"]
    context = outputs["context"]
    lifted = outputs["lifted_features"]

    # Choisissons arbitrairement:
    #
    # batch = 0
    # camera = 0
    # depth bin = 10
    # v = 3
    # u = 7
    # context channel = 20
    #
    b = 0
    n = 0
    d = 10
    v = 3
    u = 7
    c = 20

    expected = (
        depth[b, n, d, v, u]
        *
        context[b, n, c, v, u]
    )

    actual = lifted[
        b,
        n,
        d,
        v,
        u,
        c,
    ]

    assert torch.allclose(
        actual,
        expected,
        atol=1e-6,
    )


def test_lift_backward():

    model = Lift()

    model.train()

    x = torch.randn(
        1,
        2,
        512,
        8,
        22,
        requires_grad=True,
    )

    outputs = model(x)

    loss = outputs["lifted_features"].mean()

    loss.backward()

    # Le gradient doit revenir jusqu'à l'entrée.
    assert x.grad is not None

    # Et jusqu'aux paramètres du Conv1x1.
    assert model.depth_context_head.weight.grad is not None