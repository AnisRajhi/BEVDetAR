#!/usr/bin/env python3

from torch.utils.data import DataLoader

from lss_det.data.nuscenes_dataset import (
    NuScenesLSSDataset,
)

from lss_det.data.collate import (
    lss_collate_fn,
)


# ==============================================================
# CHANGE THIS
# ==============================================================

DATAROOT = "/home/anis/datasets/nuscenes"


dataset = NuScenesLSSDataset(
    dataroot=DATAROOT,
    version="v1.0-mini",
    split="train",
    verbose=True,
)


print()
print("==============================")
print("DATASET")
print("==============================")

print(
    "Samples:",
    len(dataset),
)

print(
    "Classes:",
    dataset.classes,
)

print(
    "Cameras:",
    dataset.cameras,
)


# ==============================================================
# One sample
# ==============================================================

sample = dataset[0]


print()
print("==============================")
print("SINGLE SAMPLE")
print("==============================")


for key in [
    "images",
    "intrins",
    "rots",
    "trans",
    "post_rots",
    "post_trans",
    "gt_boxes",
    "gt_labels",
]:

    print(
        f"{key:12s}:",
        sample[key].shape,
    )


print()
print(
    "Sample token:",
    sample[
        "sample_token"
    ],
)


print()
print("First GT boxes:")

print(
    sample[
        "gt_boxes"
    ][:5]
)


print()
print("First GT labels:")

print(
    sample[
        "gt_labels"
    ][:5]
)


# ==============================================================
# DataLoader
# ==============================================================

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    collate_fn=lss_collate_fn,
)


batch = next(
    iter(loader)
)


print()
print("==============================")
print("BATCH")
print("==============================")


for key in [
    "images",
    "intrins",
    "rots",
    "trans",
    "post_rots",
    "post_trans",
]:

    print(
        f"{key:12s}:",
        batch[key].shape,
    )


print()

print(
    "GT frame 0:",
    batch[
        "gt_boxes"
    ][0].shape,
)

print(
    "GT frame 1:",
    batch[
        "gt_boxes"
    ][1].shape,
)


from lss_det.targets.centerpoint_targets import (
    CenterPointTargetBuilder,
)


target_builder = CenterPointTargetBuilder(
    classes=dataset.classes,
    xbound=(-50.0, 50.0, 0.5),
    ybound=(-50.0, 50.0, 0.5),
)


targets = target_builder(
    gt_boxes=batch["gt_boxes"],
    gt_labels=batch["gt_labels"],
)


print()
print("==============================")
print("TARGETS")
print("==============================")


for key, value in targets.items():

    print(
        f"{key:12s}:",
        value.shape,
    )

print(
    "Objects used:",
    targets["mask"].sum(
        dim=1
    )
)