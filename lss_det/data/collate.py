#!/usr/bin/env python3

from typing import Dict, List

import torch


def lss_collate_fn(
    batch: List[Dict],
) -> Dict:

    if len(batch) == 0:

        raise ValueError(
            "Cannot collate empty batch."
        )

    # ==============================================================
    # Fixed-size tensors
    # ==============================================================

    output = {
        "images": torch.stack(
            [
                item["images"]
                for item in batch
            ],
            dim=0,
        ),

        "intrins": torch.stack(
            [
                item["intrins"]
                for item in batch
            ],
            dim=0,
        ),

        "rots": torch.stack(
            [
                item["rots"]
                for item in batch
            ],
            dim=0,
        ),

        "trans": torch.stack(
            [
                item["trans"]
                for item in batch
            ],
            dim=0,
        ),

        "post_rots": torch.stack(
            [
                item["post_rots"]
                for item in batch
            ],
            dim=0,
        ),

        "post_trans": torch.stack(
            [
                item["post_trans"]
                for item in batch
            ],
            dim=0,
        ),
    }

    # ==============================================================
    # Variable number of objects
    # ==============================================================

    output[
        "gt_boxes"
    ] = [
        item["gt_boxes"]
        for item in batch
    ]

    output[
        "gt_labels"
    ] = [
        item["gt_labels"]
        for item in batch
    ]

    # Metadata.

    output[
        "sample_token"
    ] = [
        item["sample_token"]
        for item in batch
    ]

    return output