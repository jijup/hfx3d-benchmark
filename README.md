# HFX3D Benchmark

Benchmark suite for semantic segmentation of urban building facade LiDAR point clouds on the HFX3D dataset.

## Dataset

**HFX3D** is a point cloud dataset of 26 urban building facades in Halifax, Nova Scotia, Canada, captured with a ZEB Horizon handheld LiDAR scanner.

| Property | Value |
|----------|-------|
| Buildings | 26 |
| Split | 18 train / 4 val / 4 test |
| Train points | ~343.7 M |
| Val points | ~68.4 M |
| Test points | ~79.7 M |
| Total points | ~491.9 M |
| Format | HDF5 (`.h5`), one file per building |
| Classes | 12 semantic classes |
| Coordinates | Mean-centered per building, metric scale preserved |
| RGB | Normalised [0, 1] (raw 16-bit / 65535) |
| Intensity | Normalised [0, 1] (p1/p99 from training split) |

### Dataset Split

| Split | Buildings |
|-------|-----------|
| Train (18) | BLD001, BLD002, BLD003, BLD004, BLD006, BLD007, BLD009, BLD010, BLD011, BLD012, BLD013, BLD014, BLD015, BLD016, BLD018, BLD019, BLD020, BLD023 |
| Val (4) | BLD008, BLD022, BLD024, BLD025 |
| Test (4) | BLD005, BLD017, BLD021, BLD026 |

Split assignment files: `data/splits/{train,val,test}.txt`

### Building Index

| ID | Name | Split |
|----|------|-------|
| BLD001 | Goldberg | train |
| BLD002 | Mona Campbell A | train |
| BLD003 | Mona Campbell B | train |
| BLD004 | Kenneth Rowe | train |
| BLD005 | Schulich School of Law | **test** |
| BLD006 | VIC Suites | train |
| BLD007 | Engineering Building | train |
| BLD008 | Mona Campbell C | val |
| BLD009 | Halifax Central Library | train |
| BLD010 | Emera Place | train |
| BLD011 | Dal Gerard Hall | train |
| BLD012 | The Margaritta | train |
| BLD013 | 1801 Hollis St 1 | train |
| BLD014 | Summit Place | train |
| BLD015 | 1801 Hollis St 2 | train |
| BLD016 | 1801 Hollis St 3 | train |
| BLD017 | 1801 Hollis St 4 | **test** |
| BLD018 | The Exchange | train |
| BLD019 | Sexton Campus 1 | train |
| BLD020 | Sexton Campus 2 | train |
| BLD021 | TD Building | **test** |
| BLD022 | Steele Ocean 1 | val |
| BLD023 | Steele Ocean 2 | train |
| BLD024 | Sobeys Inspiration Hub | val |
| BLD025 | Belmont House | val |
| BLD026 | The Avery | **test** |

### 12-Class Taxonomy

| ID | Class | Frequency |
|----|-------|-----------|
| 0 | wall | dominant |
| 1 | window | high |
| 2 | door | low |
| 3 | balcony | low |
| 4 | vegetation | medium |
| 5 | stairs | rare |
| 6 | terrain | medium |
| 7 | roof | low |
| 8 | blinds | low |
| 9 | other | medium |
| 10 | column | rare |
| 11 | arch | rare |

Class imbalance is severe. `door`, `column`, and `arch` deviate from the 80/20 split target because their points concentrate in few buildings. RGB is unreliable — ~70% of points have near-black values from the ZEB scanner.

### HDF5 File Structure

Each `.h5` file contains:

```
point_cloud/xyz        (N, 3)  float32   mean-centered per building, metric scale
point_cloud/rgb        (N, 3)  float32   [0, 1]  (raw / 65535.0)
point_cloud/intensity  (N,)    float32   [0, 1]  global p1/p99 from training split
labels/semantic        (N,)    uint8     0–11
metadata/centroid      (3,)    float64   original centroid before centering
```

All models receive the same 7D input feature vector: `[x, y, z, r, g, b, intensity]`.

## Benchmark Results

Models trained on 18 buildings, validated on 4, evaluated on 4 held-out test buildings. Results pending.

### Summary

| Model | OA | μP | μR | μF1 | mIoU |
|-------|----|----|-----|-----|------|
| PTv3 (Point Transformer V3) | — | — | — | — | — |
| PTv2 (Point Transformer V2) | — | — | — | — | — |
| DGCNN | — | — | — | — | — |
| RandLA-Net | — | — | — | — | — |
| PointNet++ (SSG) | — | — | — | — | — |
| PointNet | — | — | — | — | — |

### Per-Class IoU

| Class | PTv3 | PTv2 | DGCNN | RandLA-Net | PointNet++ | PointNet |
|-------|------|------|-------|------------|------------|---------|
| wall | — | — | — | — | — | — |
| window | — | — | — | — | — | — |
| door | — | — | — | — | — | — |
| balcony | — | — | — | — | — | — |
| vegetation | — | — | — | — | — | — |
| stairs | — | — | — | — | — | — |
| terrain | — | — | — | — | — | — |
| roof | — | — | — | — | — | — |
| blinds | — | — | — | — | — | — |
| other | — | — | — | — | — | — |
| column | — | — | — | — | — | — |
| arch | — | — | — | — | — | — |
| **mIoU** | — | — | — | — | — | — |

## Reproducing Results

See `docs/reproduction.md` for full step-by-step setup instructions.

### Prerequisites

- Alliance Canada account on `fir.alliancecan.ca`
- Apptainer SIF at `/home/<user>/pointcept.sif` (PyTorch 2.4.1, CUDA 12.1)
- Dataset at `/home/<user>/projects/def-<pi>/<user>/HFX3D_Splits/` with `train/`, `val/`, `test/` subdirs
- Set `HFX3D_DATA_ROOT` env var to your dataset path; all loaders fall back to the committed cluster path if unset

### Quick Start

```bash
# Clone this repo and each framework (see docs/reproduction.md for per-model setup)

# Submit all seed-42 training jobs at once
bash slurm/run_all_baselines.sh

# Or submit individually
sbatch slurm/pointcept/submit_ptv3_seed42.sh   # PTv3  (~24h, 800 epochs)
sbatch slurm/pointcept/submit_ptv2_seed42.sh   # PTv2  (~24h, 800 epochs)
sbatch slurm/dgcnn/submit_dgcnn_seed42.sh      # DGCNN (~15 min, 200 epochs)
sbatch slurm/randlanet/submit_randla_seed42.sh # RandLA-Net (~24h, 100 epochs)
sbatch slurm/pointnet2/submit_pn2_seed42.sh    # PointNet++ (~24h, 64 epochs)
sbatch slurm/pointnet2/submit_pn_seed42.sh     # PointNet   (~24h, 64 epochs)

# Multi-seed runs: seed123 and seed456 variants exist for all models
# e.g. sbatch slurm/pointcept/submit_ptv3_seed123.sh
```

### Framework Repos

Each framework is cloned separately. HFX3D-specific files from this repo are copied into each checkout.

| Model | Framework repo | HFX3D files | Org fork |
|-------|---------------|-------------|----------|
| PTv2 / PTv3 | [Pointcept/Pointcept](https://github.com/Pointcept/Pointcept) | `models/pointcept/` | [hfx3d-pointcept](https://github.com/Graphical-Spatial-Computing-Lab/hfx3d-pointcept) |
| DGCNN | [AnTao97/dgcnn.pytorch](https://github.com/AnTao97/dgcnn.pytorch) | `models/dgcnn/` | [hfx3d-dgcnn](https://github.com/Graphical-Spatial-Computing-Lab/hfx3d-dgcnn) |
| RandLA-Net | [QingyongHu/RandLA-Net](https://github.com/QingyongHu/RandLA-Net) | `models/randlanet/` | [hfx3d-randlanet](https://github.com/Graphical-Spatial-Computing-Lab/hfx3d-randlanet) |
| PointNet / PointNet++ | [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) | `models/pointnet2/` | [hfx3d-pointnet2](https://github.com/Graphical-Spatial-Computing-Lab/hfx3d-pointnet2) |

## Preprocessing

Raw `.las` files go through two steps before training:

```bash
# Step 1 — strip unlabelled (label-0) points and shift remaining labels down by 1
python preprocessing/preprocess.py <raw_las_folder> [--output_folder <out>]

# Step 2 — convert cleaned LAS → HDF5 with official normalization
#   input_dir must contain train/ val/ test/ subdirs with .las files
python preprocessing/las_to_hdf5.py <input_dir> <output_dir>
# writes normalization_stats.json (intensity p1/p99) alongside the HDF5 files

# Optional — class distribution analysis and charts
python preprocessing/analyse_dataset.py <las_dir> [--output_dir <out>]
```

`las_to_hdf5.py` is a two-pass script: it computes intensity p1/p99 from training files first, then converts all splits using those stats for consistent cross-building normalization.

## Infrastructure

- **Cluster:** Alliance Canada — `fir.alliancecan.ca`
- **GPUs:** NVIDIA H100 (80 GB)
- **Container:** Apptainer SIF (PyTorch 2.4.1, CUDA 12.1, h5py, scipy, torch_geometric)
- **Scheduler:** SLURM (`sbatch`)

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
│   ├── run_all_baselines.sh      # submit all seed-42 jobs in one command
│   └── evaluate_all_baselines.sh # submit evaluation jobs
├── preprocessing/
│   ├── preprocess.py    # strip label-0 points, shift labels
│   ├── las_to_hdf5.py   # LAS → HDF5 with official normalization
│   └── analyse_dataset.py
├── configs/
│   ├── palette.py       # single source of truth for all class/split colours
│   ├── label_map.json   # class name → ID mapping
│   └── scalar_fields.xml # CloudCompare colour scale
├── data/
│   └── splits/
│       ├── train.txt    # 18 building IDs
│       ├── val.txt      # 4 building IDs
│       └── test.txt     # 4 building IDs
└── docs/
    ├── reproduction.md  # step-by-step reproduction guide
    └── HFX3D_README.md  # building index and naming convention
```
