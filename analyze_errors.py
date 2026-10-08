#!/usr/bin/env python3
"""
Analyse des erreurs du détecteur sur la validation nuScenes.

    python3 analyze_errors.py

Deux parties, écrites à l'écran et dans REPORT_PATH :

1. LIMITE STRUCTURELLE DE LA TÊTE (sans modèle, toute la validation)
   On fabrique les cibles CenterPoint à partir des vraies boîtes, on les
   donne au décodeur comme si le réseau était parfait, et on mesure ce
   qu'il retrouve. Ce qui manque ici est perdu quel que soit l'entraînement
   (deux objets dans la même case, etc.). On compte aussi les objets qui
   ont un voisin de même classe dans une case adjacente : avec un vrai
   réseau (pics moins nets), ils risquent de fusionner.

2. ERREURS DU MODÈLE (NUM_FRAMES frames de validation)
   a. Devenir de chaque objet réel (prédictions avec score >= SCORE_THRESHOLD) :
      bonne classe à < 2 m | autre classe à < 2 m (confusion) |
      détecté entre 2 et 4 m (mal placé) | rien à moins de 4 m (non vu)
   b. Matrice de confusion (objets détectés à < 2 m, toutes classes)
   c. Taux d'objets non vus selon leur visibilité
   d. Erreurs de cap des objets bien classés (inversions à 180° comprises)
   e. Erreur de centre décomposée : radiale (le long du rayon = profondeur)
      et latérale, par tranche de distance
"""

import math
import os
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lss_det import config as C
from lss_det.data.collate import lss_collate_fn, model_inputs
from lss_det.data.nuscenes_dataset import NuScenesLSSDataset, release_nuscenes
from lss_det.decoding.decode_boxes import CenterPointDecoder
from lss_det.metrics.detection_metrics import DetectionEvaluator
from lss_det.models.lss_detector import LSSDetector
from lss_det.targets.centerpoint_targets import CenterPointTargetBuilder

CHECKPOINT = "checkpoints_full/best_map.pt"
REPORT_PATH = "analysis_report.txt"
NUM_FRAMES = 1500          # frames de val pour la partie 2 (réparties sur toute la val) ; 0 = toutes
SCORE_THRESHOLD = 0.3      # même seuil que visualize.py
DEVICE = "auto"            # "cpu" si un entraînement occupe déjà le GPU
RUN_ORACLE = True
RUN_MODEL = True

BANDS = ((0, 20), (20, 30), (30, 40), (40, 50))
FATES = ("bonne classe <2 m", "autre classe <2 m", "détecté 2-4 m", "non vu")


# ======================================================================
# Outils
# ======================================================================

def yaw_error(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def radial_lateral(pred_xy, gt_xy):
    """Erreur de centre décomposée le long du rayon ego -> objet (profondeur) et en travers."""
    r = gt_xy / max(np.linalg.norm(gt_xy), 1e-6)
    e = pred_xy - gt_xy
    return abs(float(e @ r)), abs(float(e[0] * r[1] - e[1] * r[0]))


def categorize_frame(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, d_ok=2.0, d_far=4.0):
    """
    Pour chaque objet réel : (devenir, classe prédite ou -1, indice de la prédiction ou -1).
    Ordre des passes (glouton par score, chaque prédiction utilisée une fois) :
      1. même classe à < d_ok   2. autre classe à < d_ok   3. toute classe à < d_far.
    """
    P, G = len(pred_boxes), len(gt_boxes)
    used = np.zeros(P, dtype=bool)
    fate = [("non vu", -1, -1)] * G
    done = np.zeros(G, dtype=bool)
    order = np.argsort(-np.asarray(pred_scores)) if P else []
    passes = ((True, d_ok, FATES[0]), (False, d_ok, FATES[1]), (False, d_far, FATES[2]))
    for same_class, dmax, name in passes:
        for i in order:
            if used[i]:
                continue
            cand = np.where(~done)[0]
            if same_class:
                cand = cand[gt_labels[cand] == pred_labels[i]]
            if cand.size == 0:
                continue
            d = np.hypot(gt_boxes[cand, 0] - pred_boxes[i, 0], gt_boxes[cand, 1] - pred_boxes[i, 1])
            j = int(np.argmin(d))
            if d[j] < dmax:
                g = cand[j]
                used[i], done[g] = True, True
                fate[g] = (name, int(pred_labels[i]), int(i))
    return fate


def perfect_predictions(targets):
    """Cibles CenterPoint -> sorties de tête « parfaites » (mêmes clés que le modèle)."""
    B, K, Nx, Ny = targets["heatmap"].shape
    logits = torch.logit(targets["heatmap"].clamp(1e-4, 1 - 1e-4))
    box = torch.zeros(B, 8, Nx * Ny)
    for b in range(B):
        m = targets["mask"][b]
        box[b][:, targets["indices"][b][m]] = targets["box_targets"][b][m].T
    box = box.reshape(B, 8, Nx, Ny)
    return {"heatmap": logits, "reg": box[:, 0:2], "center_z": box[:, 2:3], "dim": box[:, 3:6], "rot": box[:, 6:8]}


def crowding(gt_boxes, gt_labels, builder):
    """Par objet : partage-t-il sa case (0) ou une case voisine (1) avec un objet de même classe ?"""
    ix = np.floor((gt_boxes[:, 0] - builder.x_min) / builder.dx).astype(int)
    iy = np.floor((gt_boxes[:, 1] - builder.y_min) / builder.dy).astype(int)
    same_cell = np.zeros(len(gt_boxes), dtype=bool)
    neighbour = np.zeros(len(gt_boxes), dtype=bool)
    for a in range(len(gt_boxes)):
        for b in range(len(gt_boxes)):
            if a == b or gt_labels[a] != gt_labels[b]:
                continue
            cheb = max(abs(ix[a] - ix[b]), abs(iy[a] - iy[b]))
            same_cell[a] |= cheb == 0
            neighbour[a] |= cheb == 1
    return same_cell, neighbour


class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)

    def save(self, path):
        with open(path, "w") as f:
            f.write("\n".join(self.lines) + "\n")


def pct(a, b):
    return f"{100.0 * a / b:5.1f} %" if b else "    —  "


# ======================================================================
# 1. Limite structurelle de la tête
# ======================================================================

def run_oracle(ds, out):
    builder, decoder, evaluator = CenterPointTargetBuilder(), CenterPointDecoder(), DetectionEvaluator()
    n_gt, n_same, n_nb = Counter(), Counter(), Counter()
    for info in ds.infos:
        gt_boxes, gt_labels = ds.get_gt(info)
        if len(gt_boxes):
            same, nb = crowding(gt_boxes.numpy(), gt_labels.numpy(), builder)
            for lab, s, n in zip(gt_labels.tolist(), same, nb):
                n_gt[lab] += 1
                n_same[lab] += int(s)
                n_nb[lab] += int(n and not s)
        targets = builder([gt_boxes], [gt_labels])
        det = decoder(perfect_predictions(targets))[0]
        evaluator.add(info["token"], det["boxes"], det["scores"], det["labels"], gt_boxes, gt_labels)
    res = evaluator.compute()
    out("=" * 92)
    out(f"1. LIMITE STRUCTURELLE DE LA TÊTE — cibles parfaites, {len(ds.infos)} frames de validation")
    out("=" * 92)
    out(f"{'classe':<11}{'objets':>8}{'rappel max':>12}{'AP@0.5':>9}{'même case':>12}{'case voisine':>15}")
    for k, name in enumerate(C.CLASSES):
        e = res["per_class"][name]
        if e["num_gt"] == 0:
            continue
        out(f"{name:<11}{e['num_gt']:>8}{e.get('max_recall', 0):>12.3f}{e['AP'][0.5]:>9.3f}"
            f"{pct(n_same[k], n_gt[k]):>12}{pct(n_nb[k], n_gt[k]):>15}")
    out("  rappel max < 1 : objets perdus quel que soit l'entraînement (deux objets dans la même case).")
    out("  case voisine : objets à risque de fusion avec un vrai réseau, dont les pics sont moins nets.")
    out()


# ======================================================================
# 2. Erreurs du modèle
# ======================================================================

def run_model(ds, out, device):
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    saved = ckpt.get("config", {})
    for key in ("IMAGE_SIZE", "DEPTH_BOUND", "XBOUND", "YBOUND", "ZBOUND", "CLASSES"):
        if key in saved and tuple(saved[key]) != tuple(getattr(C, key)):
            raise ValueError(f"{key} du checkpoint ({saved[key]}) != config actuelle ({getattr(C, key)})")
    model = LSSDetector(camera_pretrained=False).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    decoder = CenterPointDecoder().to(device)

    n = len(ds) if NUM_FRAMES <= 0 else min(NUM_FRAMES, len(ds))
    idx = sorted(set(np.linspace(0, len(ds) - 1, n).round().astype(int).tolist()))
    token_to_info = {i["token"]: i for i in ds.infos}
    loader = DataLoader(Subset(ds, idx), batch_size=1, shuffle=False, collate_fn=lss_collate_fn,
                        num_workers=C.NUM_WORKERS)

    K = len(C.CLASSES)
    fates = defaultdict(Counter)                     # classe réelle -> devenir -> nombre
    confusion = np.zeros((K, K), dtype=int)          # réelle x prédite, objets à < 2 m
    vis_total, vis_missed = defaultdict(Counter), defaultdict(Counter)
    yaw_errs = defaultdict(list)
    radial = defaultdict(list)                       # tranche -> [(radiale, latérale)]

    for n_done, batch in enumerate(loader, 1):
        with torch.no_grad():
            det = decoder(model(**model_inputs(batch, device)))[0]
        keep = det["scores"] >= SCORE_THRESHOLD
        pb, ps, pl = (det[k][keep].cpu().numpy() for k in ("boxes", "scores", "labels"))
        gb, gl = batch["gt_boxes"][0].numpy(), batch["gt_labels"][0].numpy()
        info = token_to_info[batch["sample_token"][0]]
        anns = [a for a in ds.get_annotations(info) if ds.keep_annotation(a)]   # même ordre que get_gt

        for g, (f, plab, pi) in enumerate(categorize_frame(pb, ps, pl, gb, gl)):
            t = int(gl[g])
            fates[t][f] += 1
            if f in FATES[:2]:
                confusion[t, plab] += 1
            vis = anns[g]["visibility"] if g < len(anns) else 4
            vis_total[t][vis] += 1
            vis_missed[t][vis] += int(f == "non vu")
            if f == FATES[0]:
                yaw_errs[t].append(yaw_error(pb[pi, 6], gb[g, 6]))
            if f in FATES[:3] and plab == t:
                dist = math.hypot(gb[g, 0], gb[g, 1])
                for lo, hi in BANDS:
                    if lo <= dist < hi:
                        radial[(lo, hi)].append(radial_lateral(pb[pi, :2], gb[g, :2]))
        if n_done % 200 == 0:
            print(f"  ... {n_done}/{len(idx)} frames")

    out("=" * 92)
    out(f"2. ERREURS DU MODÈLE — {CHECKPOINT}, {len(idx)} frames de validation, score >= {SCORE_THRESHOLD}")
    out("=" * 92)
    out("a. Devenir de chaque objet réel")
    out(f"{'classe':<11}{'objets':>8}" + "".join(f"{f:>20}" for f in FATES))
    for k, name in enumerate(C.CLASSES):
        tot = sum(fates[k].values())
        if tot:
            out(f"{name:<11}{tot:>8}" + "".join(f"{pct(fates[k][f], tot):>20}" for f in FATES))
    out()
    out("b. Confusion (objets détectés à < 2 m) : lignes = classe réelle, colonnes = classe prédite (%)")
    out(f"{'':<11}" + "".join(f"{c[:10]:>11}" for c in C.CLASSES))
    for k, name in enumerate(C.CLASSES):
        row = confusion[k]
        if row.sum():
            out(f"{name:<11}" + "".join(f"{100 * v / row.sum():>10.1f}%" for v in row))
    out()
    out("c. Part d'objets NON VUS selon leur visibilité (nuScenes : 1 = 0-40 %, ..., 4 = 80-100 %)")
    out(f"{'classe':<11}" + "".join(f"{lab:>18}" for lab in ("vis 0-40 %", "40-60 %", "60-80 %", "80-100 %")))
    for k, name in enumerate(C.CLASSES):
        if sum(vis_total[k].values()):
            out(f"{name:<11}" + "".join(
                f"{pct(vis_missed[k][v], vis_total[k][v]) + f' /{vis_total[k][v]:>5}':>18}" for v in (1, 2, 3, 4)))
    out()
    out("d. Erreurs de cap des objets bien classés à < 2 m")
    out(f"{'classe':<11}{'n':>7}{'< 15°':>10}{'15-45°':>10}{'45-135°':>10}{'> 135° (inversé)':>19}{'médiane':>10}")
    for k, name in enumerate(C.CLASSES):
        e = np.degrees(np.array(yaw_errs[k]))
        if e.size:
            parts = [(e < 15).mean(), ((e >= 15) & (e < 45)).mean(), ((e >= 45) & (e < 135)).mean(), (e >= 135).mean()]
            out(f"{name:<11}{e.size:>7}" + "".join(f"{100 * p:>9.1f}%" for p in parts[:3])
                + f"{100 * parts[3]:>18.1f}%{np.median(e):>9.1f}°")
    out()
    out("e. Erreur de centre (bonne classe, < 4 m) : radiale = profondeur, latérale = en travers du rayon")
    out(f"{'distance':<10}{'n':>7}{'médiane radiale':>18}{'médiane latérale':>19}{'radiale / latérale':>21}")
    for lo, hi in BANDS:
        v = np.array(radial[(lo, hi)])
        if len(v):
            r, l = np.median(v[:, 0]), np.median(v[:, 1])
            out(f"{f'{lo}-{hi} m':<10}{len(v):>7}{r:>16.2f} m{l:>17.2f} m{r / max(l, 1e-6):>20.1f}×")
    out()


def main():
    device = torch.device("cuda" if DEVICE == "auto" and torch.cuda.is_available()
                          else ("cpu" if DEVICE == "auto" else DEVICE))
    ds = NuScenesLSSDataset(split="val", training=False)
    release_nuscenes()
    out = Report()
    if RUN_ORACLE:
        run_oracle(ds, out)
    if RUN_MODEL:
        run_model(ds, out, device)
    out.save(REPORT_PATH)
    print(f"Rapport écrit dans {REPORT_PATH}")


if __name__ == "__main__":
    main()
