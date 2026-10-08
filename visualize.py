#!/usr/bin/env python3
"""
Visualisation des détections sur la validation nuScenes.

    python3 visualize.py

Pour chaque frame choisie, deux images dans OUTPUT_DIR :

  frame_XXXXX.png  6 caméras (boîtes prédites en couleur, vérité terrain en
                   pointillés blancs) + vue de dessus (lidar en gris, heatmap
                   du réseau en rouge, vérité terrain en pointillés noirs,
                   objets MANQUÉS en pointillés rouges, prédictions en couleur
                   avec un trait vers l'avant de l'objet).
  depth_XXXXX.png  pour DEPTH_CAMERAS : image vue par le réseau, profondeur
                   prédite, profondeur lidar, erreur absolue.

Mode "scene" : toutes les frames d'une scène, plus un GIF (2 images/s, le
rythme des keyframes nuScenes).

IMAGE_SIZE dans config.py doit être celui de l'entraînement du checkpoint.
Si un entraînement tourne déjà sur le GPU, mettre DEVICE = "cpu" (plus lent,
quelques secondes par frame, mais sans perturber l'entraînement).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from lss_det import config as C
from lss_det.data.collate import lss_collate_fn, model_inputs
from lss_det.data.nuscenes_dataset import NuScenesLSSDataset, release_nuscenes
from lss_det.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from lss_det.decoding.decode_boxes import CenterPointDecoder
from lss_det.models.lss_detector import LSSDetector
from lss_det.visualization.draw import (CLASS_COLORS, GT_COLOR_IMAGE, draw_box_on_image, draw_bev,
                                        legend_handles, match_frame)

CHECKPOINT = "checkpoints_full/best_map.pt"
OUTPUT_DIR = "visualizations"
MODE = "spread"            # "spread" : NUM_FRAMES frames réparties sur la val | "scene" : une scène entière
NUM_FRAMES = 12
SCENE_POSITION = 0         # mode "scene" : 0 = première scène de val, 1 = deuxième, ...
MAKE_GIF = True            # mode "scene"
SCORE_THRESHOLD = 0.3      # prédictions affichées (et comptées) à partir de ce score
DEPTH_CAMERAS = ("CAM_FRONT", "CAM_BACK")
DEVICE = "auto"            # "auto", "cuda" ou "cpu"
IMAGE_SCALE = 0.5          # images caméra affichées à 800x450
CAMERA_LAYOUT = [["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
                 ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]]


# ======================================================================
# Chargement
# ======================================================================

def load_model(device):
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    saved = ckpt.get("config", {})
    for key in ("IMAGE_SIZE", "DEPTH_BOUND", "XBOUND", "YBOUND", "ZBOUND", "CLASSES"):
        if key in saved and tuple(saved[key]) != tuple(getattr(C, key)):
            raise ValueError(f"{key} du checkpoint ({saved[key]}) != config actuelle ({getattr(C, key)})")
    model = LSSDetector(camera_pretrained=False, return_intermediates=True).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt.get("epoch")


def frame_timestamp(info):
    """Horodatage (µs) lu dans le nom du fichier lidar nuScenes."""
    try:
        return int(os.path.basename(info["lidar_path"]).split("__")[-1].split(".")[0])
    except (ValueError, IndexError):
        return 0


def select_frames(ds):
    if MODE == "scene":
        scenes = list(dict.fromkeys(i["scene_token"] for i in ds.infos))
        token = scenes[SCENE_POSITION % len(scenes)]
        idx = [k for k, i in enumerate(ds.infos) if i["scene_token"] == token]
        return sorted(idx, key=lambda k: frame_timestamp(ds.infos[k]))
    n = min(NUM_FRAMES, len(ds))
    return sorted(set(np.linspace(0, len(ds) - 1, n).round().astype(int).tolist()))


# ======================================================================
# Figures
# ======================================================================

def render_overview(ds, idx, item, det, heat, gt_found, pred_tp, path):
    info = ds.infos[idx]
    gt_boxes, gt_labels = item["gt_boxes"].numpy(), item["gt_labels"].numpy()
    fig = plt.figure(figsize=(26, 9.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.05], wspace=0.03, hspace=0.08)

    for r, row in enumerate(CAMERA_LAYOUT):
        for c, cam in enumerate(row):
            ax = fig.add_subplot(gs[r, c])
            k = C.CAMERAS.index(cam)
            img = Image.open(info["cams"][k]["path"]).convert("RGB")
            W, H = img.size
            ax.imshow(img.resize((int(W * IMAGE_SCALE), int(H * IMAGE_SCALE))))
            R, t, K = ds.get_camera_to_reference(info, k)
            for box in gt_boxes:
                draw_box_on_image(ax, box, R, t, K, (W, H), IMAGE_SCALE, GT_COLOR_IMAGE, "--", 1.0)
            for box, lab in zip(det["boxes"], det["labels"]):
                draw_box_on_image(ax, box, R, t, K, (W, H), IMAGE_SCALE, CLASS_COLORS[C.CLASSES[int(lab)]], "-", 1.6)
            ax.set_xlim(0, W * IMAGE_SCALE)
            ax.set_ylim(H * IMAGE_SCALE, 0)
            ax.set_title(cam, fontsize=10)
            ax.axis("off")

    ax = fig.add_subplot(gs[:, 3])
    draw_bev(ax, C.CLASSES, gt_boxes, gt_labels, gt_found, det["boxes"], det["scores"], det["labels"],
             heatmap=heat, lidar_points=ds.get_lidar_points_reference(info), bev_range=C.XBOUND[1])
    n_tp, n_pred, n_gt = int(pred_tp.sum()), len(pred_tp), len(gt_boxes)
    ax.set_title(f"Vue de dessus · vérité terrain {n_gt} · prédictions {n_pred} "
                 f"(justes {n_tp}, fausses {n_pred - n_tp}) · manquées {n_gt - int(gt_found.sum())}", fontsize=10)
    ax.legend(handles=legend_handles(C.CLASSES), loc="lower left", fontsize=7, framealpha=0.85)
    fig.suptitle(f"Validation, frame {idx} · sample {info['token'][:8]} · score ≥ {SCORE_THRESHOLD} · "
                 f"appariement : même classe, centre à moins de 2 m", fontsize=12)
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def render_depth(item, depth_probs, path):
    dmin, _, step = C.DEPTH_BOUND
    centers = dmin + (np.arange(C.NUM_DEPTH_BINS) + 0.5) * step
    mean, std = np.array(IMAGENET_MEAN), np.array(IMAGENET_STD)
    cams = [c for c in DEPTH_CAMERAS if c in C.CAMERAS]
    depth_cmap = plt.get_cmap("turbo").copy()
    depth_cmap.set_bad("#e0e0e0")                       # cellules sans point lidar : gris clair
    err_cmap = plt.get_cmap("magma").copy()
    err_cmap.set_bad("#e0e0e0")
    fig, axes = plt.subplots(len(cams), 4, figsize=(24, 3.1 * len(cams)), squeeze=False, constrained_layout=True)
    for row, cam in enumerate(cams):
        k = C.CAMERAS.index(cam)
        img = (item["images"][k].numpy().transpose(1, 2, 0) * std + mean).clip(0, 1)
        H, W = img.shape[:2]
        probs = depth_probs[k]                                         # [D, Hf, Wf]
        pred = (probs * centers[:, None, None]).sum(0)
        bins = item["depth_bins"][k].numpy()
        labelled = bins >= 0
        lidar = np.ma.masked_where(~labelled, dmin + (bins + 0.5) * step)
        err = np.ma.masked_where(~labelled, np.abs(pred - (dmin + (bins + 0.5) * step)))
        acc = f"{(np.abs(probs.argmax(0) - bins)[labelled] <= 1).mean():.2f}" if labelled.any() else "—"
        ext = (0, W, H, 0)
        panels = [
            (img, None, f"{cam} : image vue par le réseau ({W}×{H})"),
            (pred, depth_cmap, "profondeur prédite (espérance)"),
            (lidar, depth_cmap, f"profondeur lidar ({labelled.mean():.0%} des cellules ; gris = aucun point)"),
            (err, err_cmap, f"erreur absolue · dAcc ±1 m = {acc}"),
        ]
        for col, (data, cmap, title) in enumerate(panels):
            ax = axes[row, col]
            if cmap is None:
                ax.imshow(data, extent=ext)
            else:
                vmax = 10 if col == 3 else 60
                im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax, extent=ext, interpolation="nearest")
                if col in (2, 3):
                    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="m")
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(path, dpi=80)
    plt.close(fig)


# ======================================================================
# Boucle
# ======================================================================

def main():
    device = torch.device("cuda" if DEVICE == "auto" and torch.cuda.is_available()
                          else ("cpu" if DEVICE == "auto" else DEVICE))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model, epoch = load_model(device)
    decoder = CenterPointDecoder().to(device)
    ds = NuScenesLSSDataset(split="val", training=False)
    release_nuscenes()
    frames = select_frames(ds)
    print(f"Checkpoint {CHECKPOINT} (epoch {epoch}) · {len(frames)} frames · device {device} · sortie {OUTPUT_DIR}/")

    overview_paths = []
    for n, idx in enumerate(frames, 1):
        item = ds[idx]
        batch = lss_collate_fn([item])
        with torch.no_grad():
            preds = model(**model_inputs(batch, device))
            det = decoder(preds)[0]
        keep = det["scores"] >= SCORE_THRESHOLD
        det = {k: v[keep].cpu().numpy() for k, v in det.items()}
        heat = torch.sigmoid(preds["heatmap"][0].float()).amax(0).cpu().numpy()
        pred_tp, gt_found = match_frame(det["boxes"], det["scores"], det["labels"],
                                        item["gt_boxes"].numpy(), item["gt_labels"].numpy())

        p1 = os.path.join(OUTPUT_DIR, f"frame_{idx:05d}.png")
        render_overview(ds, idx, item, det, heat, gt_found, pred_tp, p1)
        render_depth(item, preds["depth_probs"][0].float().cpu().numpy(), os.path.join(OUTPUT_DIR, f"depth_{idx:05d}.png"))
        overview_paths.append(p1)
        print(f"  [{n}/{len(frames)}] frame {idx} : vérité terrain {len(gt_found)}, prédictions {len(pred_tp)} "
              f"(justes {int(pred_tp.sum())}), manquées {len(gt_found) - int(gt_found.sum())}")

    if MODE == "scene" and MAKE_GIF and overview_paths:
        imgs = [Image.open(p).convert("RGB") for p in overview_paths]
        w = 1400
        imgs = [im.resize((w, int(im.height * w / im.width))) for im in imgs]
        gif = os.path.join(OUTPUT_DIR, f"scene_{SCENE_POSITION:03d}.gif")
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=500, loop=0)
        print(f"GIF : {gif}")


if __name__ == "__main__":
    main()
