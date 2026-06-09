# Adapted from Pointcept (https://github.com/Pointcept/Pointcept)
# Config structure and PTv2 hyperparameters derived from their ScanNet/S3DIS configs.
# Modified for HFX3D: in_channels=7, num_classes=12, HFX3DDataset loader.

import os
_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 6  # bs: total bs across 2x H100 (3/GPU)
mix_prob = 0.8
empty_cache = False
enable_amp = True

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PT-v2m2",
        in_channels=7,  # xyz (3) + rgb (3) + intensity (1) = 7
        num_classes=12,  # HFX3D has 12 classes
        patch_embed_depth=1,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=8,
        enc_depths=(2, 2, 6, 2),
        enc_channels=(96, 192, 384, 512),
        enc_groups=(12, 24, 48, 64),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(48, 96, 192, 384),
        dec_groups=(6, 12, 24, 48),
        dec_neighbours=(16, 16, 16, 16),
        grid_sizes=(0.06, 0.15, 0.375, 0.9375),  # metric scale (6cm→15cm→37.5cm→93.75cm)
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        enable_checkpoint=False,
        unpool_backend="map",
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

# scheduler settings
epoch = 200
seed = 42
optimizer = dict(type="AdamW", lr=0.005, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings
dataset_type = "HFX3DDataset"
data_root = os.environ.get("HFX3D_DATA_ROOT", "/path/to/HFX3D_Splits")  # set HFX3D_DATA_ROOT env var

# max_points_per_building=None → full cloud flows into GridSample+SphereCrop.
# The old 500k random cap discarded ~97% of each building and broke test coverage.
dataset_params = dict(
    max_points_per_building=None,
    sample_method="random",
    compute_normals=False,
)

data = dict(
    num_classes=12,
    ignore_index=-1,
    names=[
        "wall", "window", "door", "balcony", "vegetation", "stairs",
        "terrain", "roof", "blinds", "other", "column", "arch",
    ],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        max_points_per_building=dataset_params["max_points_per_building"],
        sample_method=dataset_params["sample_method"],
        compute_normals=dataset_params["compute_normals"],
        transform=[
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train", return_min_coord=True),
            dict(type="SphereCrop", point_max=100000, mode="random"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "segment"), feat_keys=("coord", "color", "intensity")),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        max_points_per_building=dataset_params["max_points_per_building"],
        sample_method=dataset_params["sample_method"],
        compute_normals=dataset_params["compute_normals"],
        # Val uses a local SphereCrop (same as train) for checkpoint selection.
        # A full-cloud forward OOMs on 19M-pt buildings. Full-scene eval is done
        # at test time via tools/test_fullscene_hfx3d.py (KNN-crop voting).
        transform=[
            dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train"),
            dict(type="SphereCrop", point_max=100000, mode="random"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "segment"), feat_keys=("coord", "color", "intensity")),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        max_points_per_building=dataset_params["max_points_per_building"],
        sample_method=dataset_params["sample_method"],
        compute_normals=dataset_params["compute_normals"],
        transform=[],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="test"),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "index"), feat_keys=("coord", "color", "intensity")),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)
