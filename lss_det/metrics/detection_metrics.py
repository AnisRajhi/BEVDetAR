#!/usr/bin/env python3
"""
Métriques de détection 3D "à la nuScenes", dans le repère ego.

Reproduit l'algorithme officiel (nuscenes.eval.detection) :
  - appariement glouton par score décroissant, sur la distance des
    centres en BEV (pas d'IoU), seuils {0.5, 1, 2, 4} m ;
  - AP = aire normalisée sous la courbe précision/rappel interpolée
    sur 101 points, avec min_recall = min_precision = 0.1 ;
  - erreurs TP au seuil 2 m : ATE (m), ASE (1 - IoU alignée),
    AOE (rad), moyennées comme dans le devkit (moyenne cumulée
    interpolée entre rappel 0.1 et le rappel max atteint).

Non inclus (le modèle ne les prédit pas) : AVE (vitesse), AAE
(attributs), donc pas de NDS officiel. Ajout : rappel par tranche de
distance, pour voir OÙ le modèle échoue.

Les GT passées ici doivent avoir subi le même filtrage que l'éval
officielle (0 point lidar+radar retirés, portée par classe) : c'est ce
que fait le dataset v2.
"""

import math
from collections import defaultdict
from typing import Dict, Sequence

import numpy as np

from lss_det import config as C

MIN_RECALL = 0.1
MIN_PRECISION = 0.1


def _cummean(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    return np.cumsum(x) / np.arange(1, x.size + 1)


def _yaw_diff(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d)


def _scale_error(dp: np.ndarray, dg: np.ndarray) -> float:
    inter = np.prod(np.minimum(dp, dg))
    union = np.prod(dp) + np.prod(dg) - inter
    return float(1.0 - inter / max(union, 1e-9))


class DetectionEvaluator:
    def __init__(
        self,
        classes: Sequence[str] = C.CLASSES,
        class_range: Dict[str, float] = C.CLASS_RANGE,
        dist_ths: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
        tp_dist_th: float = 2.0,
        distance_bands: Sequence = ((0, 20), (20, 30), (30, 40), (40, 50)),
    ):
        self.classes = list(classes)
        self.class_range = dict(class_range)
        self.dist_ths = tuple(dist_ths)
        self.tp_dist_th = float(tp_dist_th)
        self.distance_bands = tuple(distance_bands)
        self.reset()

    def reset(self):
        self.preds = defaultdict(list)     # class -> [(sample, score, box)]
        self.gts = defaultdict(dict)       # class -> sample -> [M,7]
        self.num_samples = 0

    @staticmethod
    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    def add(self, sample_id, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels):
        pb, ps, pl = self._np(pred_boxes), self._np(pred_scores), self._np(pred_labels)
        gb, gl = self._np(gt_boxes), self._np(gt_labels)
        self.num_samples += 1
        for k, name in enumerate(self.classes):
            g = gb[gl == k] if gb.size else np.zeros((0, 7))
            self.gts[name][sample_id] = g.reshape(-1, 7)
            if pb.size == 0:
                continue
            sel = pl == k
            boxes, scores = pb[sel].reshape(-1, 7), ps[sel]
            # Filtre officiel : prédictions au-delà de la portée de la classe
            dist = np.hypot(boxes[:, 0], boxes[:, 1])
            keep = dist <= self.class_range[name]
            for box, score in zip(boxes[keep], scores[keep]):
                self.preds[name].append((sample_id, float(score), box))

    # --------------------------------------------------------------

    def _accumulate(self, name: str, dist_th: float):
        gts = self.gts[name]
        npos = sum(v.shape[0] for v in gts.values())
        preds = sorted(self.preds[name], key=lambda t: -t[1])
        taken = {sid: np.zeros(v.shape[0], dtype=bool) for sid, v in gts.items()}

        tp, fp, conf = [], [], []
        errs = {"trans": [], "scale": [], "orient": [], "conf": []}
        for sid, score, box in preds:
            g = gts.get(sid, np.zeros((0, 7)))
            best, best_d = -1, np.inf
            if g.shape[0]:
                d = np.hypot(g[:, 0] - box[0], g[:, 1] - box[1])
                d[taken[sid]] = np.inf
                best = int(np.argmin(d))
                best_d = d[best]
            if best >= 0 and best_d < dist_th:
                taken[sid][best] = True
                tp.append(1); fp.append(0)
                gbox = g[best]
                errs["trans"].append(best_d)
                errs["scale"].append(_scale_error(box[3:6], gbox[3:6]))
                errs["orient"].append(_yaw_diff(box[6], gbox[6]))
                errs["conf"].append(score)
            else:
                tp.append(0); fp.append(1)
            conf.append(score)

        return npos, np.array(tp), np.array(fp), np.array(conf), errs, taken

    @staticmethod
    def _ap_and_curves(npos, tp, fp, conf):
        rec_interp = np.linspace(0, 1, 101)
        if npos == 0 or tp.size == 0 or tp.sum() == 0:
            return 0.0, rec_interp, np.zeros(101), np.zeros(101)
        tp_c, fp_c = np.cumsum(tp).astype(float), np.cumsum(fp).astype(float)
        prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
        rec = tp_c / float(npos)
        prec_i = np.interp(rec_interp, rec, prec, right=0)
        conf_i = np.interp(rec_interp, rec, conf, right=0)

        p = prec_i[round(100 * MIN_RECALL) + 1:] - MIN_PRECISION
        p[p < 0] = 0
        ap = float(np.mean(p)) / (1.0 - MIN_PRECISION)
        return ap, rec_interp, prec_i, conf_i

    def compute(self) -> Dict:
        res = {"per_class": {}, "num_samples": self.num_samples}
        aps_all, tp_errs = [], {"ATE": [], "ASE": [], "AOE": []}
        band_hits = {b: [0, 0] for b in self.distance_bands}

        for name in self.classes:
            npos = sum(v.shape[0] for v in self.gts[name].values())
            entry = {"num_gt": int(npos), "num_pred": len(self.preds[name]), "AP": {}}
            if npos == 0:
                entry["mAP"] = float("nan")
                res["per_class"][name] = entry
                continue

            for th in self.dist_ths:
                n, tp, fp, conf, errs, taken = self._accumulate(name, th)
                ap, _, _, conf_i = self._ap_and_curves(n, tp, fp, conf)
                entry["AP"][th] = ap
                aps_all.append(ap)

                if th == self.tp_dist_th:
                    first = round(100 * MIN_RECALL) + 1
                    nz = np.nonzero(conf_i)[0]
                    last = int(nz[-1]) if nz.size else 0
                    for key, metric in (("trans", "ATE"), ("scale", "ASE"), ("orient", "AOE")):
                        if last < first or len(errs[key]) == 0:
                            val = 1.0
                        else:
                            tmp = _cummean(np.array(errs[key]))
                            e_conf = np.array(errs["conf"])
                            interp = np.interp(conf_i[::-1], e_conf[::-1], tmp[::-1])[::-1]
                            val = float(np.mean(interp[first:last + 1]))
                        entry[metric] = val
                        tp_errs[metric].append(val)
                    entry["max_recall"] = float(tp.sum() / max(n, 1))

                    for sid, g in self.gts[name].items():
                        d = np.hypot(g[:, 0], g[:, 1])
                        for (lo, hi) in self.distance_bands:
                            sel = (d >= lo) & (d < hi)
                            band_hits[(lo, hi)][0] += int(taken[sid][sel].sum())
                            band_hits[(lo, hi)][1] += int(sel.sum())

            entry["mAP"] = float(np.mean(list(entry["AP"].values())))
            res["per_class"][name] = entry

        res["mAP"] = float(np.mean(aps_all)) if aps_all else 0.0
        for metric, vals in tp_errs.items():
            res[metric] = float(np.mean(vals)) if vals else 1.0
        res["recall_by_distance@2m"] = {
            f"{lo}-{hi}m": (h / t if t else float("nan"), t) for (lo, hi), (h, t) in band_hits.items()
        }
        return res


def format_report(res: Dict) -> str:
    lines = [f"mAP = {res['mAP']:.4f} | ATE = {res['ATE']:.3f} m | ASE = {res['ASE']:.3f} | AOE = {res['AOE']:.3f} rad"]
    lines.append(f"{'classe':<11} {'#GT':>5} {'#pred':>6}  AP@0.5  AP@1   AP@2   AP@4   mAP    ATE    ASE    AOE   recall")
    for name, e in res["per_class"].items():
        if e["num_gt"] == 0:
            lines.append(f"{name:<11} {0:>5} {e['num_pred']:>6}  (aucune GT)")
            continue
        ap = e["AP"]
        lines.append(
            f"{name:<11} {e['num_gt']:>5} {e['num_pred']:>6}  "
            + "  ".join(f"{ap[t]:.3f}" for t in sorted(ap))
            + f"  {e['mAP']:.3f}  {e.get('ATE', 1):.3f}  {e.get('ASE', 1):.3f}  {e.get('AOE', 1):.3f}  {e.get('max_recall', 0):.3f}"
        )
    bands = "  ".join(f"{k}: {v[0]:.2f} (n={v[1]})" for k, v in res["recall_by_distance@2m"].items())
    lines.append(f"rappel @2m par distance : {bands}")
    return "\n".join(lines)
