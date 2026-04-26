_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 12  # bs: total bs in all gpus
mix_prob = 0.8
empty_cache = False
enable_amp = True

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PT-v2m2",
        in_channels=10,  # xyz (3) + rgb (3) + normal (3) + intensity (1) = 10
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
        grid_sizes=(0.01, 0.02, 0.04, 0.08), # CHANGED: scaled for normalized [-1,1] coords (0.06, 0.15, 0.375, 0.9375),  # x3, x2.5, x2.5, x2.5
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        enable_checkpoint=False,
        unpool_backend="map",  # map / interp
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

# scheduler settings
epoch = 900
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
data_root = "/home/abdiaziz/projects/def-jijup/abdiaziz/WACV_Splits"

# Dataset parameters for direct HDF5 loading
dataset_params = dict(
    max_points_per_building=500000,  # Sample 500k points per building per epoch
    sample_method='random',
    compute_normals=False,  # Not used - loads from HDF5 if available, else dummy [0,0,1]
)

data = dict(
    num_classes=12,
    ignore_index=-1,
    names=[
        "wall",
        "window",
        "door",
        "balcony",
        "vegetation",
        "stairs",
        "terrain",
        "roof",
        "blinds",
        "other",
        "column",
        "arch",
    ],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        max_points_per_building=dataset_params["max_points_per_building"],
        sample_method=dataset_params["sample_method"],
        compute_normals=dataset_params["compute_normals"],
        transform=[
            # Note: XYZ already centered and unit-sphere normalized from LAS converter
            # Note: RGB already normalized to [0,1] from LAS converter (65535.0)
            # Note: Intensity already min-max normalized per file
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            # Building facades are mostly vertical, so reduce rotation augmentation
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
            dict(
                type="GridSample",
                grid_size=0.002,  # 0.5cm voxel size (maximum detail for small objects)
                hash_type="fnv",
                mode="train",
                return_min_coord=True,
            ),
            dict(type="SphereCrop", point_max=100000, mode="random"),
            # Removed: CenterShift (data already centered)
            # Removed: NormalizeColor (data already normalized)
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=("coord", "color", "normal", "intensity"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="test",  # HFX3D uses "test" split for validation
        data_root=data_root,
        max_points_per_building=dataset_params["max_points_per_building"],
        sample_method=dataset_params["sample_method"],
        compute_normals=dataset_params["compute_normals"],
        transform=[
            # Data already normalized from LAS converter
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.002,  # 0.5cm voxel size
                hash_type="fnv",
                mode="train",
                return_min_coord=True,
                return_inverse=True,
            ),
            # Removed: CenterShift (data already centered)
            # Removed: NormalizeColor (data already normalized)
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment", "origin_segment", "inverse"),
                feat_keys=("coord", "color", "normal", "intensity"),
            ),
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
        transform=[
            # Data already normalized from LAS converter
            # Removed: CenterShift (data already centered)
            # Removed: NormalizeColor (data already normalized)
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.002,  # 0.5cm voxel size
                hash_type="fnv",
                mode="test",
            ),
            crop=None,
            post_transform=[
                # Removed: CenterShift (data already centered)
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "index"),
                    feat_keys=("coord", "color", "normal", "intensity"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)
