#!/usr/bin/env python3
"""
Outils de dessin pour visualiser les détections sur nuScenes.

Repère : ego de référence (X avant, Y gauche, Z haut), boîtes
[x, y, z, l, w, h, yaw] avec z au centre de la boîte.

Vue de dessus : l'avant du véhicule est en HAUT, sa gauche à GAUCHE.
On trace donc (−y, x).
"""

import math
from typing import Dict, List, Sequence

import numpy as np

CLASS_COLORS = {
    "car": "#1f77b4",
    "truck": "#ff7f0e",
    "bus": "#9467bd",
    "pedestrian": "#2ca02c",
    "bicycle": "#17becf",
    "motorcycle": "#e377c2",
}
MISSED_COLOR = "#d62728"
GT_COLOR_IMAGE = "white"
GT_COLOR_BEV = "black"

# 12 arêtes d'une boîte : 0-3 = bas, 4-7 = haut (même ordre)
BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def box_corners(box: Sequence[float]) -> np.ndarray:
    """Retourne les 8 coins [8, 3] d'une boîte [x, y, z, l, w, h, yaw]."""
    x, y, z, l, w, h, yaw = (float(v) for v in box[:7])
    dx = np.array([l, l, -l, -l]) / 2.0          # avant-gauche, avant-droit, arrière-droit, arrière-gauche
    dy = np.array([w, -w, -w, w]) / 2.0
    c, s = math.cos(yaw), math.sin(yaw)
    px = x + c * dx - s * dy
    py = y + s * dx + c * dy
    bottom = np.stack([px, py, np.full(4, z - h / 2)], axis=1)
    top = np.stack([px, py, np.full(4, z + h / 2)], axis=1)
    return np.concatenate([bottom, top], axis=0)


def project_to_image(points_ego: np.ndarray, R_cam_ref: np.ndarray, t_cam_ref: np.ndarray, K: np.ndarray):
    """
    points_ego [N, 3] -> (uv [N, 2] dans l'image ORIGINALE, profondeur [N]).
    R_cam_ref, t_cam_ref : caméra -> ego de référence (comme dans le dataset).
    """
    p_cam = (points_ego - t_cam_ref) @ R_cam_ref          # = R^T (p - t)
    z = p_cam[:, 2]
    uv = (p_cam @ K.T)[:, :2] / np.maximum(z, 1e-6)[:, None]
    return uv, z


def match_frame(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, dist_th: float = 2.0):
    """
    Appariement glouton par score, même classe, distance des centres < dist_th
    (comme la métrique). Retourne (pred_is_tp [P], gt_is_found [G]).
    """
    pred_boxes, gt_boxes = np.asarray(pred_boxes).reshape(-1, 7), np.asarray(gt_boxes).reshape(-1, 7)
    pred_is_tp = np.zeros(len(pred_boxes), dtype=bool)
    gt_found = np.zeros(len(gt_boxes), dtype=bool)
    for i in np.argsort(-np.asarray(pred_scores)):
        same = np.where((np.asarray(gt_labels) == pred_labels[i]) & ~gt_found)[0]
        if same.size == 0:
            continue
        d = np.hypot(gt_boxes[same, 0] - pred_boxes[i, 0], gt_boxes[same, 1] - pred_boxes[i, 1])
        j = int(np.argmin(d))
        if d[j] < dist_th:
            pred_is_tp[i] = True
            gt_found[same[j]] = True
    return pred_is_tp, gt_found


def draw_box_on_image(ax, box, R, t, K, image_wh, scale: float, color, linestyle="-", linewidth=1.5):
    """Dessine une boîte 3D projetée. Les arêtes derrière la caméra sont ignorées."""
    corners = box_corners(box)
    uv, z = project_to_image(corners, R, t, K)
    if (z > 0.5).sum() < 2:
        return False
    W, H = image_wh
    if not ((uv[:, 0] > -W) & (uv[:, 0] < 2 * W) & (uv[:, 1] > -H) & (uv[:, 1] < 2 * H) & (z > 0.5)).any():
        return False
    uv = uv * scale
    for a, b in BOX_EDGES:
        if z[a] > 0.5 and z[b] > 0.5:
            ax.plot([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]], color=color, linestyle=linestyle, linewidth=linewidth)
    # face avant marquée d'une croix (indique le cap)
    for a, b in ((0, 5), (1, 4)):
        if z[a] > 0.5 and z[b] > 0.5:
            ax.plot([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]], color=color, linestyle=linestyle,
                    linewidth=max(0.8, linewidth * 0.6))
    return True


def _bev_xy(points):
    """Points ego [..., >=2] -> coordonnées de tracé (−y, x)."""
    return -points[..., 1], points[..., 0]


def draw_bev(ax, classes: List[str], gt_boxes, gt_labels, gt_found, pred_boxes, pred_scores, pred_labels,
             heatmap=None, lidar_points=None, bev_range: float = 51.2):
    """Vue de dessus : lidar, heatmap du réseau, vérité terrain, prédictions."""
    if heatmap is not None:
        img = np.ma.masked_less(heatmap[::-1, ::-1], 0.05)        # lignes = x décroissant, colonnes = y décroissant
        ax.imshow(img, extent=(-bev_range, bev_range, -bev_range, bev_range), cmap="Reds", vmin=0.0, vmax=1.0,
                  alpha=0.55, origin="upper", zorder=1)
    if lidar_points is not None and len(lidar_points):
        p = lidar_points
        keep = (np.abs(p[:, 0]) < bev_range) & (np.abs(p[:, 1]) < bev_range) & (p[:, 2] > -2.0) & (p[:, 2] < 3.0)
        u, v = _bev_xy(p[keep])
        ax.scatter(u, v, s=0.15, c="#9a9a9a", linewidths=0, zorder=0)
    for r in (10, 20, 30, 40, 50):
        ax.add_patch(_circle(r))
        ax.text(0.7, r + 0.5, f"{r} m", fontsize=6, color="#777777", zorder=2)

    for box, lab, found in zip(np.asarray(gt_boxes).reshape(-1, 7), gt_labels, gt_found):
        c = box_corners(box)[:4]
        u, v = _bev_xy(np.vstack([c, c[:1]]))
        ax.plot(u, v, color=GT_COLOR_BEV if found else MISSED_COLOR, linestyle="--",
                linewidth=1.0 if found else 1.6, zorder=3)
    for box, score, lab in zip(np.asarray(pred_boxes).reshape(-1, 7), pred_scores, pred_labels):
        color = CLASS_COLORS.get(classes[int(lab)], "blue")
        c = box_corners(box)[:4]
        u, v = _bev_xy(np.vstack([c, c[:1]]))
        ax.plot(u, v, color=color, linewidth=1.4, zorder=4)
        front = (c[0] + c[1]) / 2.0                                  # milieu de la face avant
        cu, cv = _bev_xy(np.array([box[:2], front[:2]]))
        ax.plot(cu, cv, color=color, linewidth=1.0, zorder=4)

    # véhicule ego (4,1 m x 1,8 m) et son cap
    ego = box_corners([0.0, 0.0, 0.0, 4.1, 1.8, 1.5, 0.0])[:4]
    u, v = _bev_xy(np.vstack([ego, ego[:1]]))
    ax.fill(u, v, color="#333333", zorder=5)
    ax.annotate("", xy=(0, 4.5), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5), zorder=5)
    ax.set_xlim(-bev_range, bev_range)
    ax.set_ylim(-bev_range, bev_range)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _circle(r):
    from matplotlib.patches import Circle
    return Circle((0, 0), r, fill=False, color="#cccccc", linewidth=0.6, zorder=0)


def legend_handles(classes: List[str]):
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color=CLASS_COLORS[c], lw=2, label=c) for c in classes]
    h.append(Line2D([0], [0], color=GT_COLOR_BEV, lw=1, ls="--", label="vérité terrain (trouvée)"))
    h.append(Line2D([0], [0], color=MISSED_COLOR, lw=1.6, ls="--", label="vérité terrain MANQUÉE"))
    return h
