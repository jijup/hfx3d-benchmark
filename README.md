# HFX3D Benchmark

Semantic segmentation benchmark for urban building facade LiDAR point clouds. Six baselines evaluated on the HFX3D dataset — 26 buildings in Halifax, NS, scanned with a ZEB Horizon handheld LiDAR.

## Repository Structure

```
hfx3d-benchmark/
├── models/
│   ├── pointcept/       # HFX3DDataset class + PTv2/PTv3 configs
│   ├── dgcnn/           # training, eval, and full-scene inference scripts
│   ├── randlanet/       # dataset adapter, model, training, and eval scripts
│   └── pointnet2/       # dataset loader + training/eval scripts (PointNet & PointNet++)
├── slurm/
│   ├── pointcept/       # PTv2/PTv3 SLURM scripts (seed 42/123/456)
│   ├── dgcnn/           # DGCNN SLURM scripts (seed 42/123/456)
│   ├── randlanet/       # RandLA-Net SLURM scripts (seed 42/123/456) + eval
│   ├── pointnet2/       # PointNet/PointNet++ SLURM scripts (seed 42/123/456)
│   ├── run_all_baselines.sh
│   └── evaluate_all_baselines.sh
├── preprocessing/
│   ├── preprocess.py    # strip label-0 points, shift labels
│   ├── las_to_hdf5.py   # LAS → HDF5 with official normalization
│   └── analyse_dataset.py
├── configs/
│   ├── palette.py       # single source of truth for all class/split colours
│   ├── label_map.json
│   └── scalar_fields.xml
├── data/splits/         # train.txt / val.txt / test.txt
└── docs/
    ├── reproduction.md
    ├── HFX3D_README.md  # building index + naming convention
    └── label_taxonomy.md
```

## Dataset

| Property | Value |
|----------|-------|
| Buildings | 26 |
| Split | 18 train / 4 val / 4 test (building-level) |
| Total points | ~491.9 M |
| Classes | 12 |
| Coordinates | Mean-centered per building, metric scale preserved |
| RGB | [0, 1] (raw 16-bit / 65535) |
| Intensity | [0, 1] (p1/p99 computed from training split) |

Each building is stored as an HDF5 file with keys `point_cloud/{xyz,rgb,intensity}`, `labels/semantic`, and `metadata/centroid`. All models receive the same 7D input: `[x, y, z, r, g, b, intensity]`.

- Building names and IDs → [`docs/HFX3D_README.md`](docs/HFX3D_README.md)
- Split assignments and point counts → [`data/splits/split.md`](data/splits/split.md)
- 12-class taxonomy and notes on class imbalance → [`docs/label_taxonomy.md`](docs/label_taxonomy.md)

## Benchmark Results

Trained on 18 buildings, evaluated on 4 held-out test buildings. Results pending.

| Model | OA | mIoU | μF1 |
|-------|----|----|-----|
| PTv3 (Point Transformer V3) | — | — | — |
| PTv2 (Point Transformer V2) | — | — | — |
| DGCNN | — | — | — |
| RandLA-Net | — | — | — |
| PointNet++ (SSG) | — | — | — |
| PointNet | — | — | — |

## Reproducing Results

Full setup instructions (framework cloning, file placement, dataset path configuration) are in [`docs/reproduction.md`](docs/reproduction.md).

```bash
# Submit all seed-42 training jobs at once
bash slurm/run_all_baselines.sh

# Or individually — seed123/456 variants exist for all models
sbatch slurm/pointcept/submit_ptv3_seed42.sh   # PTv3  (~24h, 800 epochs)
sbatch slurm/pointcept/submit_ptv2_seed42.sh   # PTv2  (~24h, 800 epochs)
sbatch slurm/dgcnn/submit_dgcnn_seed42.sh      # DGCNN (~15 min, 200 epochs)
sbatch slurm/randlanet/submit_randla_seed42.sh # RandLA-Net (~24h, 100 epochs)
sbatch slurm/pointnet2/submit_pn2_seed42.sh    # PointNet++ (~24h, 64 epochs)
sbatch slurm/pointnet2/submit_pn_seed42.sh     # PointNet   (~24h, 64 epochs)
```

Set `HFX3D_DATA_ROOT` to your `HFX3D_Splits/` path before training; all dataset loaders read this env var.

### Framework Repos

| Model | Upstream | Org fork |
|-------|----------|----------|
| PTv2 / PTv3 | [Pointcept/Pointcept](https://github.com/Pointcept/Pointcept) | [hfx3d-pointcept](https://github.com/Graphical-Spatial-Computing-Lab/hfx3d-pointcept) |
| DGCNN | [AnTao97/dgcnn.pytorch](https://github.com/AnTao97/dgcnn.pytorch) | [hfx3d-dgcnn](https://github.com/Graphical-Spatial-Computing-Lab/hfx3d-dgcnn) |
| RandLA-Net | [QingyongHu/RandLA-Net](https://github.com/QingyongHu/RandLA-Net) | [hfx3d-randlanet](https://github.com/Graphical-Spatial-Computing-Lab/hfx3d-randlanet) |
| PointNet / PointNet++ | [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) | [hfx3d-pointnet2](https://github.com/Graphical-Spatial-Computing-Lab/hfx3d-pointnet2) |

## Preprocessing

```bash
# Step 1 — strip unlabelled points from raw LAS, shift labels down by 1
python preprocessing/preprocess.py <raw_las_folder>

# Step 2 — convert to HDF5 with official normalization
#   input_dir must contain train/ val/ test/ subdirs
python preprocessing/las_to_hdf5.py <input_dir> <output_dir>
```

`las_to_hdf5.py` computes intensity p1/p99 from training files in the first pass, then normalises all splits with those stats.

## Infrastructure

- **Cluster:** Alliance Canada — `fir.alliancecan.ca`
- **GPUs:** NVIDIA H100 (80 GB)
- **Container:** Apptainer SIF (PyTorch 2.4.1, CUDA 12.1)
- **Scheduler:** SLURM
