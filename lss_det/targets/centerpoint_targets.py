#!/usr/bin/env python3

import math
from typing import Dict, List, Sequence, Tuple

import torch


# ==============================================================
# Gaussian utilities
# ==============================================================

def gaussian2d(
    radius: int,
    sigma: float = None,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Crée une Gaussian 2D carrée.

    radius = 2

    donne une matrice:
        5 x 5

    car:
        diameter = 2*radius + 1
    """

    diameter = 2 * radius + 1

    if sigma is None:
        sigma = diameter / 6.0

    coords = torch.arange(
        -radius,
        radius + 1,
        device=device,
        dtype=dtype,
    )

    xx, yy = torch.meshgrid(
        coords,
        coords,
        indexing="ij",
    )

    gaussian = torch.exp(
        -(xx * xx + yy * yy)
        / (2.0 * sigma * sigma)
    )

    return gaussian


def gaussian_radius(
    length_cells: float,
    width_cells: float,
    min_overlap: float = 0.1,
) -> float:
    """
    Rayon inspiré de CenterNet / CenterPoint.

    La taille de l'objet est exprimée en cellules BEV.

    Exemple:
        voiture:
            l = 4.2 m
            w = 1.8 m

        résolution = 0.5 m

        length_cells = 8.4
        width_cells  = 3.6

    Plus l'objet est grand, plus la Gaussian peut être large.
    """

    height = float(length_cells)
    width = float(width_cells)

    # ----------------------------------------------------------
    # Solution 1
    # ----------------------------------------------------------

    a1 = 1.0
    b1 = height + width
    c1 = (
        width
        * height
        * (1.0 - min_overlap)
        / (1.0 + min_overlap)
    )

    sq1 = math.sqrt(
        max(
            0.0,
            b1 * b1 - 4.0 * a1 * c1,
        )
    )

    r1 = (b1 + sq1) / 2.0

    # ----------------------------------------------------------
    # Solution 2
    # ----------------------------------------------------------

    a2 = 4.0
    b2 = 2.0 * (height + width)
    c2 = (
        (1.0 - min_overlap)
        * width
        * height
    )

    sq2 = math.sqrt(
        max(
            0.0,
            b2 * b2 - 4.0 * a2 * c2,
        )
    )

    r2 = (b2 + sq2) / 2.0

    # ----------------------------------------------------------
    # Solution 3
    # ----------------------------------------------------------

    a3 = 4.0 * min_overlap
    b3 = -2.0 * min_overlap * (
        height + width
    )
    c3 = (
        (min_overlap - 1.0)
        * width
        * height
    )

    sq3 = math.sqrt(
        max(
            0.0,
            b3 * b3 - 4.0 * a3 * c3,
        )
    )

    r3 = (
        b3 + sq3
    ) / 2.0

    return min(
        r1,
        r2,
        r3,
    )


def draw_gaussian(
    heatmap: torch.Tensor,
    center_x: int,
    center_y: int,
    radius: int,
) -> None:
    """
    Dessine une Gaussian sur UNE heatmap:

        heatmap: [Nx, Ny]

    Attention à notre convention:

        premier axe spatial = X
        deuxième axe spatial = Y

    donc:

        heatmap[ix, iy]
    """

    Nx, Ny = heatmap.shape

    if not (
        0 <= center_x < Nx
        and
        0 <= center_y < Ny
    ):
        return

    gaussian = gaussian2d(
        radius=radius,
        device=heatmap.device,
        dtype=heatmap.dtype,
    )

    # Limites disponibles autour du centre.

    left = min(
        center_x,
        radius,
    )

    right = min(
        Nx - center_x - 1,
        radius,
    )

    top = min(
        center_y,
        radius,
    )

    bottom = min(
        Ny - center_y - 1,
        radius,
    )

    # Partie de la heatmap à modifier.

    hm_x0 = center_x - left
    hm_x1 = center_x + right + 1

    hm_y0 = center_y - top
    hm_y1 = center_y + bottom + 1

    # Partie correspondante de la Gaussian.

    g_x0 = radius - left
    g_x1 = radius + right + 1

    g_y0 = radius - top
    g_y1 = radius + bottom + 1

    heatmap_patch = heatmap[
        hm_x0:hm_x1,
        hm_y0:hm_y1,
    ]

    gaussian_patch = gaussian[
        g_x0:g_x1,
        g_y0:g_y1,
    ]

    # Si plusieurs objets / Gaussians se chevauchent,
    # on garde la valeur maximale.

    torch.maximum(
        heatmap_patch,
        gaussian_patch,
        out=heatmap_patch,
    )


# ==============================================================
# CenterPoint Target Builder
# ==============================================================

class CenterPointTargetBuilder:
    """
    Construit les targets de notre Detection Head.

    --------------------------------------------------------------
    GT INPUT
    --------------------------------------------------------------

    gt_boxes:
        liste de B tensors

        gt_boxes[b]:
            [M_b, 7]

        avec:

            0 = x
            1 = y
            2 = z
            3 = length
            4 = width
            5 = height
            6 = yaw


    gt_labels:
        liste de B tensors

        gt_labels[b]:
            [M_b]

        labels compris entre:

            0 et K-1


    --------------------------------------------------------------
    OUTPUT
    --------------------------------------------------------------

    heatmap:
        [B,K,Nx,Ny]

    indices:
        [B,max_objects]

    mask:
        [B,max_objects]

    offset:
        [B,max_objects,2]

    center_z:
        [B,max_objects,1]

    dim:
        [B,max_objects,3]

    rot:
        [B,max_objects,2]

    labels:
        [B,max_objects]
    """

    def __init__(
        self,
        classes: Sequence[str],
        xbound: Tuple[float, float, float] = (
            -50.0,
            50.0,
            0.5,
        ),
        ybound: Tuple[float, float, float] = (
            -50.0,
            50.0,
            0.5,
        ),
        max_objects: int = 500,
        gaussian_overlap: float = 0.1,
        min_radius: int = 2,
    ):
        self.classes = list(classes)

        self.num_classes = len(
            self.classes
        )

        self.x_min = float(
            xbound[0]
        )

        self.x_max = float(
            xbound[1]
        )

        self.dx = float(
            xbound[2]
        )

        self.y_min = float(
            ybound[0]
        )

        self.y_max = float(
            ybound[1]
        )

        self.dy = float(
            ybound[2]
        )

        self.Nx = int(
            round(
                (
                    self.x_max
                    - self.x_min
                )
                / self.dx
            )
        )

        self.Ny = int(
            round(
                (
                    self.y_max
                    - self.y_min
                )
                / self.dy
            )
        )

        self.max_objects = int(
            max_objects
        )

        self.gaussian_overlap = float(
            gaussian_overlap
        )

        self.min_radius = int(
            min_radius
        )

    def __call__(
        self,
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        if len(gt_boxes) != len(gt_labels):
            raise ValueError(
                "gt_boxes and gt_labels "
                "must have same batch size."
            )

        B = len(gt_boxes)

        if B == 0:
            raise ValueError(
                "Batch cannot be empty."
            )

        # On utilise le device des GT.

        device = gt_boxes[0].device

        dtype = gt_boxes[0].dtype

        # ==========================================================
        # 1. Allocation
        # ==========================================================

        heatmap = torch.zeros(
            B,
            self.num_classes,
            self.Nx,
            self.Ny,
            dtype=dtype,
            device=device,
        )

        indices = torch.zeros(
            B,
            self.max_objects,
            dtype=torch.long,
            device=device,
        )

        mask = torch.zeros(
            B,
            self.max_objects,
            dtype=torch.bool,
            device=device,
        )

        offset = torch.zeros(
            B,
            self.max_objects,
            2,
            dtype=dtype,
            device=device,
        )

        center_z = torch.zeros(
            B,
            self.max_objects,
            1,
            dtype=dtype,
            device=device,
        )

        dim = torch.zeros(
            B,
            self.max_objects,
            3,
            dtype=dtype,
            device=device,
        )

        rot = torch.zeros(
            B,
            self.max_objects,
            2,
            dtype=dtype,
            device=device,
        )

        labels_out = torch.zeros(
            B,
            self.max_objects,
            dtype=torch.long,
            device=device,
        )

        # ==========================================================
        # 2. Parcourir le batch
        # ==========================================================

        for b in range(B):

            boxes_b = gt_boxes[b]

            labels_b = gt_labels[b]

            if boxes_b.ndim != 2:
                raise ValueError(
                    "Each gt_boxes[b] must "
                    "have shape [M,7]."
                )

            if boxes_b.shape[-1] != 7:
                raise ValueError(
                    "Expected GT boxes format "
                    "[x,y,z,l,w,h,yaw]."
                )

            if labels_b.ndim != 1:
                raise ValueError(
                    "Each gt_labels[b] must "
                    "have shape [M]."
                )

            if boxes_b.shape[0] != labels_b.shape[0]:
                raise ValueError(
                    "Number of boxes and labels "
                    "does not match."
                )

            target_index = 0

            # ======================================================
            # 3. Chaque objet GT
            # ======================================================

            for obj_idx in range(
                boxes_b.shape[0]
            ):

                if target_index >= self.max_objects:
                    break

                box = boxes_b[obj_idx]

                label = int(
                    labels_b[obj_idx].item()
                )

                if not (
                    0
                    <= label
                    < self.num_classes
                ):
                    raise ValueError(
                        f"Invalid class label {label}."
                    )

                x = float(
                    box[0].item()
                )

                y = float(
                    box[1].item()
                )

                z = box[2]

                length = box[3]

                width = box[4]

                height = box[5]

                yaw = box[6]

                # --------------------------------------------------
                # 4. Metric XY -> continuous grid coordinates
                # --------------------------------------------------

                gx = (
                    x - self.x_min
                ) / self.dx

                gy = (
                    y - self.y_min
                ) / self.dy

                # --------------------------------------------------
                # 5. Cell center
                # --------------------------------------------------

                ix = int(
                    math.floor(gx)
                )

                iy = int(
                    math.floor(gy)
                )

                # Objet hors BEV:
                #
                # on l'ignore complètement.

                if not (
                    0 <= ix < self.Nx
                    and
                    0 <= iy < self.Ny
                ):
                    continue

                # Dimensions invalides:
                #
                # une box avec l<=0 ou w<=0
                # n'a pas de sens.

                if (
                    float(length.item()) <= 0.0
                    or
                    float(width.item()) <= 0.0
                    or
                    float(height.item()) <= 0.0
                ):
                    continue

                # --------------------------------------------------
                # 6. Gaussian radius
                # --------------------------------------------------

                length_cells = (
                    float(length.item())
                    / self.dx
                )

                width_cells = (
                    float(width.item())
                    / self.dy
                )

                radius = gaussian_radius(
                    length_cells=length_cells,
                    width_cells=width_cells,
                    min_overlap=self.gaussian_overlap,
                )

                radius = max(
                    self.min_radius,
                    int(radius),
                )

                # --------------------------------------------------
                # 7. Draw class heatmap
                # --------------------------------------------------

                draw_gaussian(
                    heatmap=heatmap[
                        b,
                        label,
                    ],
                    center_x=ix,
                    center_y=iy,
                    radius=radius,
                )

                # --------------------------------------------------
                # 8. Linear index du centre
                # --------------------------------------------------
                #
                # Notre spatial layout:
                #
                # [Nx,Ny]
                #
                # donc:
                #
                # linear_index =
                # ix * Ny + iy
                #

                linear_index = (
                    ix * self.Ny
                    + iy
                )

                indices[
                    b,
                    target_index,
                ] = linear_index

                # --------------------------------------------------
                # 9. Offset sub-cell
                # --------------------------------------------------

                offset[
                    b,
                    target_index,
                    0,
                ] = gx - ix

                offset[
                    b,
                    target_index,
                    1,
                ] = gy - iy

                # --------------------------------------------------
                # 10. Z
                # --------------------------------------------------

                center_z[
                    b,
                    target_index,
                    0,
                ] = z

                # --------------------------------------------------
                # 11. Dimensions
                # --------------------------------------------------

                dim[
                    b,
                    target_index,
                    0,
                ] = length

                dim[
                    b,
                    target_index,
                    1,
                ] = width

                dim[
                    b,
                    target_index,
                    2,
                ] = height

                # --------------------------------------------------
                # 12. Rotation
                # --------------------------------------------------

                rot[
                    b,
                    target_index,
                    0,
                ] = torch.sin(
                    yaw
                )

                rot[
                    b,
                    target_index,
                    1,
                ] = torch.cos(
                    yaw
                )

                # --------------------------------------------------
                # 13. Metadata
                # --------------------------------------------------

                labels_out[
                    b,
                    target_index,
                ] = label

                mask[
                    b,
                    target_index,
                ] = True

                target_index += 1

        return {
            "heatmap": heatmap,
            "indices": indices,
            "mask": mask,
            "offset": offset,
            "center_z": center_z,
            "dim": dim,
            "rot": rot,
            "labels": labels_out,
        }