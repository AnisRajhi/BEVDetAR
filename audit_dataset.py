#!/usr/bin/env python3
"""
Audit du dataset AVANT d'entraîner.

    python3 audit_dataset.py

Répond à des questions qui ne nécessitent aucun réseau :

  1. Combien de cibles de l'ANCIEN pipeline (v1) étaient impossibles ?
       - objet totalement occulté (0 point lidar + radar)
       - au-delà de la portée d'évaluation de sa classe
       - centre visible dans aucune image (après resize/crop v1)
       - visible, mais profondeur caméra hors [4, 45) m
  2. Les cibles v2 sont-elles toutes atteignables par la géométrie v2 ?
  3. Distribution des distances / visibilités par classe.
  4. Diversité réelle : scènes, frames par scène.
  5. Couverture des labels de profondeur lidar par caméra (v2).
  6. Focales par caméra (pourquoi le Lift doit connaître la caméra).

"Visible" = le CENTRE de la boîte se projette dans l'image recadrée,
devant la caméra. C'est une approximation (un objet proche coupé par le
crop peut être partiellement visible avec un centre hors image).
"""

import math
from collections import Counter, defaultdict

import numpy as np

from lss_det import config as C
from lss_det.data.depth_targets import build_depth_target, transform_points
from lss_det.data.nuscenes_dataset import NuScenesLSSDataset, release_nuscenes
from lss_det.data.transforms import LSSImageAugmentation

SPLITS = ("train", "val")
AUDIT_LIDAR = True        # charge les nuages lidar (un peu plus lent)

# Ancienne configuration, pour mesurer ce que la v1 demandait au réseau.
OLD = {
    "image_size": (128, 352),
    "bot_pct": 0.11,
    "depth": (4.0, 45.0),
    "bev_half": 50.0,
}
NEW = {
    "image_size": C.IMAGE_SIZE,
    "bot_pct": C.VAL_BOT_PCT,
    "depth": (C.DEPTH_BOUND[0], C.DEPTH_BOUND[1]),
    "bev_half": min(C.XBOUND[1], C.YBOUND[1]),
}


def make_projector(cfg):
    aug = LSSImageAugmentation(cfg["image_size"], training=False, val_bot_pct=cfg["bot_pct"])
    W, H = C.ORIGINAL_IMAGE_SIZE[1], C.ORIGINAL_IMAGE_SIZE[0]
    post_rot, post_trans = aug.compute_post_transform((W, H), aug.sample_params(W, H, None))
    return post_rot.numpy().astype(np.float64), post_trans.numpy().astype(np.float64)


def center_status(center_ref, cams, cfg, post):
    """
    Retourne (visible_dans_une_image, profondeur_ok_dans_une_image_qui_le_voit).
    cams : liste de (R_cam_ref, t_cam_ref, K).
    """
    post_rot, post_trans = post
    fH, fW = cfg["image_size"]
    dmin, dmax = cfg["depth"]
    visible = depth_ok = False
    for R, t, K in cams:
        p = R.T @ (center_ref - t)
        if p[2] <= 0.1:
            continue
        uv = (K @ p)[:2] / p[2]
        uv = post_rot[:2, :2] @ uv + post_trans[:2]
        if -0.5 <= uv[0] < fW - 0.5 and -0.5 <= uv[1] < fH - 0.5:
            visible = True
            if dmin <= p[2] < dmax:
                depth_ok = True
    return visible, depth_ok


def pct(a, b):
    return f"{100.0 * a / b:5.1f} %" if b else "   -  "


def audit_split(split, nusc=None):
    ds = NuScenesLSSDataset(split=split, training=False, nusc=nusc, use_lidar_depth=False)
    post_old, post_new = make_projector(OLD), make_projector(NEW)

    old = defaultdict(Counter)        # classe -> compteurs v1
    new = defaultdict(Counter)        # classe -> compteurs v2
    distances = defaultdict(list)
    visibility = defaultdict(Counter)
    frames_per_scene = Counter()
    focal = defaultdict(set)
    depth_cells = defaultdict(list)

    for info in ds.infos:
        frames_per_scene[info["scene_token"]] += 1
        cams = []
        for k, cam in enumerate(ds.cameras):
            R, t, K = ds.get_camera_to_reference(info, k)
            cams.append((R, t, K))
            focal[cam].add(round(float(K[0, 0])))

        for a in ds.get_annotations(info):
            name = a["name"]
            c = np.asarray(a["box"][:3])
            distances[name].append(a["distance"])
            visibility[name][a["visibility"]] += 1

            # ---------------- ce que la v1 entraînait ----------------
            if abs(c[0]) < OLD["bev_half"] and abs(c[1]) < OLD["bev_half"]:
                o = old[name]
                o["targets"] += 1
                empty = a["num_pts"] <= 0
                far = a["distance"] > C.CLASS_RANGE[name]
                vis, dok = center_status(c, cams, OLD, post_old)
                o["empty"] += empty
                o["beyond_eval_range"] += far
                o["not_in_any_image"] += not vis
                o["visible_but_depth_out"] += vis and not dok
                o["learnable"] += (not empty) and (not far) and dok

            # ---------------- ce que la v2 entraîne ----------------
            if ds.keep_annotation(a):
                n = new[name]
                n["targets"] += 1
                vis, dok = center_status(c, cams, NEW, post_new)
                n["not_in_any_image"] += not vis
                n["visible_but_depth_out"] += vis and not dok
                n["reachable"] += dok

        if AUDIT_LIDAR:
            lidar = ds.get_lidar_points_reference(info)
            for (R, t, K), cam in zip(cams, ds.cameras):
                pts_cam = transform_points(lidar, R.T, -R.T @ t)
                bins, _ = build_depth_target(pts_cam, K, post_new[0], post_new[1],
                                             C.IMAGE_SIZE, C.DOWNSAMPLE, C.DEPTH_BOUND)
                depth_cells[cam].append(float((bins >= 0).mean()))

    # ======================== rapport ========================
    print("=" * 78)
    fps = sorted(frames_per_scene.values())
    detail = (", ".join(str(v) for v in frames_per_scene.values()) if len(fps) <= 12
              else f"min {fps[0]}, médiane {fps[len(fps) // 2]}, max {fps[-1]}")
    print(f"SPLIT {split} : {len(ds.infos)} frames, {len(frames_per_scene)} scènes ({detail} frames/scène)")
    print("=" * 78)

    print("\n[1] Cibles de la v1 (toutes les annotations dans ±50 m)")
    print(f"{'classe':<11}{'cibles':>8}{'0 point':>10}{'hors portée':>13}{'hors image':>12}"
          f"{'prof. hors':>12}{'apprenables':>13}")
    tot = Counter()
    for name in C.CLASSES:
        o = old[name]
        tot.update(o)
        print(f"{name:<11}{o['targets']:>8}{pct(o['empty'], o['targets']):>10}{pct(o['beyond_eval_range'], o['targets']):>13}"
              f"{pct(o['not_in_any_image'], o['targets']):>12}{pct(o['visible_but_depth_out'], o['targets']):>12}"
              f"{pct(o['learnable'], o['targets']):>13}")
    print(f"{'TOTAL':<11}{tot['targets']:>8}{pct(tot['empty'], tot['targets']):>10}{pct(tot['beyond_eval_range'], tot['targets']):>13}"
          f"{pct(tot['not_in_any_image'], tot['targets']):>12}{pct(tot['visible_but_depth_out'], tot['targets']):>12}"
          f"{pct(tot['learnable'], tot['targets']):>13}")
    print("  (les colonnes se recouvrent : un objet peut être vide ET hors portée)")

    print("\n[2] Cibles de la v2 (après filtrage)")
    print(f"{'classe':<11}{'cibles':>8}{'hors image':>12}{'prof. hors':>12}{'atteignables':>14}")
    tot = Counter()
    for name in C.CLASSES:
        n = new[name]
        tot.update(n)
        print(f"{name:<11}{n['targets']:>8}{pct(n['not_in_any_image'], n['targets']):>12}"
              f"{pct(n['visible_but_depth_out'], n['targets']):>12}{pct(n['reachable'], n['targets']):>14}")
    print(f"{'TOTAL':<11}{tot['targets']:>8}{pct(tot['not_in_any_image'], tot['targets']):>12}"
          f"{pct(tot['visible_but_depth_out'], tot['targets']):>12}{pct(tot['reachable'], tot['targets']):>14}")

    print("\n[3] Distances (m) et visibilité, toutes annotations des classes suivies")
    print(f"{'classe':<11}{'n':>6}{'p10':>7}{'p50':>7}{'p90':>7}{'max':>7}   vis 0-40 / 40-60 / 60-80 / 80-100 %")
    for name in C.CLASSES:
        d = np.asarray(distances[name]) if distances[name] else np.zeros(1)
        v = visibility[name]
        nv = sum(v.values())
        print(f"{name:<11}{len(distances[name]):>6}{np.percentile(d, 10):>7.1f}{np.percentile(d, 50):>7.1f}"
              f"{np.percentile(d, 90):>7.1f}{d.max():>7.1f}   "
              + " / ".join(pct(v[k], nv).strip() for k in (1, 2, 3, 4)))

    print("\n[4] Focales par caméra (px) — une conv 1x1 partagée ne peut pas les deviner")
    for cam in ds.cameras:
        print(f"  {cam:<17} fx = {sorted(focal[cam])}")

    if AUDIT_LIDAR:
        print(f"\n[5] Couverture des labels de profondeur lidar (v2, {C.FEATURE_SIZE[0]}x{C.FEATURE_SIZE[1]} cellules)")
        for cam in ds.cameras:
            vals = depth_cells[cam]
            print(f"  {cam:<17} {100 * np.mean(vals):5.1f} % des cellules labellisées en moyenne")
    print()

    return {"old": old, "new": new}


def main(nusc_by_split=None):
    print("Configuration v2 :", C.summary())
    results = {}
    for split in SPLITS:
        nusc = nusc_by_split.get(split) if nusc_by_split else None
        results[split] = audit_split(split, nusc=nusc)
    release_nuscenes()
    return results


if __name__ == "__main__":
    main()
