#!/usr/bin/env python3
"""
Supervision explicite de la profondeur (BEVDepth).

Problème de l'ancien pipeline
-----------------------------
La distribution de profondeur n'était apprise qu'à travers la loss de
détection : quelques dizaines de centres d'objets par frame, en BEV,
après un splat non différentiable par rapport à la position. Le réseau
peut alors produire des distributions qui "marchent" sur les scènes de
train (en étalant la masse là où se trouvent les objets du train) sans
apprendre une profondeur géométrique transférable.

Ici on projette le nuage LIDAR_TOP de la frame dans chaque caméra et on
fabrique, pour chaque cellule de la feature map (stride 16), le bin de
profondeur du point lidar le plus proche. Cela donne des milliers de
labels denses et géométriquement exacts par frame, au lieu de ~20.

Convention de cellule (identique à Geometry) :
    cellule j  <->  pixels [16 j, 16 j + 15]  (centres de pixel entiers)
    centre     =    16 j + 7.5
"""

from typing import Tuple

import numpy as np


def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """points [N,3] ; retourne R @ p + t pour chaque point."""
    return points @ R.T + t


def build_depth_target(
    points_cam: np.ndarray,
    intrinsic: np.ndarray,
    post_rot: np.ndarray,
    post_trans: np.ndarray,
    image_size: Tuple[int, int],
    downsample: int,
    depth_bound: Tuple[float, float, float],
):
    """
    points_cam : [N,3] points dans le repère optique caméra
                 (X droite, Y bas, Z avant).

    Retourne
    --------
    depth_bins : [Hf, Wf] int64, indice de bin, -1 = pas de label
    depth_min  : [Hf, Wf] float32, profondeur (m), 0 = pas de label
    """
    H, W = image_size
    Hf, Wf = H // downsample, W // downsample
    dmin, dmax, dstep = depth_bound
    D = int(round((dmax - dmin) / dstep))

    depth_bins = np.full((Hf, Wf), -1, dtype=np.int64)
    depth_min = np.zeros((Hf, Wf), dtype=np.float32)

    if points_cam.shape[0] == 0:
        return depth_bins, depth_min

    z = points_cam[:, 2]
    keep = z > 0.1
    p = points_cam[keep]
    z = z[keep]
    if p.shape[0] == 0:
        return depth_bins, depth_min

    uv = (p @ intrinsic.T)[:, :2] / z[:, None]
    uv = uv @ post_rot[:2, :2].T + post_trans[:2]

    cx = np.floor((uv[:, 0] + 0.5) / downsample).astype(np.int64)
    cy = np.floor((uv[:, 1] + 0.5) / downsample).astype(np.int64)
    inside = (cx >= 0) & (cx < Wf) & (cy >= 0) & (cy < Hf)
    cx, cy, z = cx[inside], cy[inside], z[inside]
    if z.shape[0] == 0:
        return depth_bins, depth_min

    # Profondeur minimale par cellule : au bord d'un objet, c'est
    # l'avant-plan qui est visible.
    flat = cy * Wf + cx
    order = np.lexsort((z, flat))          # tri par cellule puis profondeur
    flat, z = flat[order], z[order]
    first = np.ones_like(flat, dtype=bool)
    first[1:] = flat[1:] != flat[:-1]
    flat, z = flat[first], z[first]

    depth_min.reshape(-1)[flat] = z
    bins = np.floor((z - dmin) / dstep).astype(np.int64)
    valid = (bins >= 0) & (bins < D)
    depth_bins.reshape(-1)[flat[valid]] = bins[valid]
    return depth_bins, depth_min
