#!/usr/bin/env python3

import torch

from lss_det.models.lss_detector import (
    LSSDetector,
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device,
)


B = 1
N = 4
K = 6


model = LSSDetector(
    num_classes=K,
    camera_pretrained=False,
    return_intermediates=True,
).to(device)

model.eval()


# ==============================================================
# Images
# ==============================================================

images = torch.randn(
    B,
    N,
    3,
    128,
    352,
    device=device,
)


# ==============================================================
# Intrinsics
# ==============================================================

K_matrix = torch.tensor(
    [
        [200.0,   0.0, 176.0],
        [  0.0, 200.0,  64.0],
        [  0.0,   0.0,   1.0],
    ],
    device=device,
)

intrins = (
    K_matrix
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


# ==============================================================
# Camera optical -> ego
# ==============================================================

R = torch.tensor(
    [
        [0.0,  0.0,  1.0],
        [-1.0, 0.0,  0.0],
        [0.0, -1.0,  0.0],
    ],
    device=device,
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


trans = torch.zeros(
    B,
    N,
    3,
    device=device,
)

trans[..., 2] = 1.2


post_rots = (
    torch.eye(
        3,
        device=device,
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
    device=device,
)


# ==============================================================
# Forward
# ==============================================================

with torch.no_grad():

    outputs = model(
        images,
        intrins,
        rots,
        trans,
        post_rots,
        post_trans,
    )


print(
    "\n============================"
)
print(
    "FULL LSS DETECTOR FORWARD"
)
print(
    "============================\n"
)


for key, value in outputs.items():

    if isinstance(
        value,
        dict,
    ):

        print(
            f"{key}:"
        )

        for (
            sub_key,
            sub_value,
        ) in value.items():

            print(
                f"  {sub_key:12s}: "
                f"{tuple(sub_value.shape)}"
            )

    else:

        print(
            f"{key:18s}: "
            f"{tuple(value.shape)}"
        )
