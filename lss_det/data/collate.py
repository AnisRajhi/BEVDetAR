#!/usr/bin/env python3

from typing import Dict, List

import torch

STACKED_KEYS = (
    "images", "intrins", "rots", "trans", "post_rots", "post_trans", "bda", "depth_bins",
)
LIST_KEYS = ("gt_boxes", "gt_labels", "sample_token")


def lss_collate_fn(batch: List[Dict]) -> Dict:
    if not batch:
        raise ValueError("Cannot collate empty batch.")
    out = {k: torch.stack([item[k] for item in batch], dim=0) for k in STACKED_KEYS}
    for k in LIST_KEYS:
        out[k] = [item[k] for item in batch]
    return out


MODEL_INPUT_KEYS = ("images", "intrins", "rots", "trans", "post_rots", "post_trans", "bda")


def model_inputs(batch: Dict, device) -> Dict:
    return {k: batch[k].to(device, non_blocking=True) for k in MODEL_INPUT_KEYS}
