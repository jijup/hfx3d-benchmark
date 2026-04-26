# HFX3D Benchmark

Benchmark suite for semantic segmentation of urban building facade LiDAR point clouds on the HFX3D dataset.

## Dataset

**HFX3D** is a point cloud dataset of urban building facades in Halifax, Nova Scotia, Canada, captured using a ZEB Horizon handheld LiDAR scanner.

| Property | Value |
|----------|-------|
| Total buildings | 23 |
| Benchmark split | 13 train / 5 test |
| Train points | ~268M |
| Test points | ~67M |
| Format | HDF5 (`.h5`), one file per building |
| Classes | 12 semantic classes |
| Intensity | Normalised [0, 1] |
| Coordinates | Unit-sphere normalised per building |

### 12-Class Taxonomy

| ID | Class | Description |
|----|-------|-------------|
| 0 | wall | Exterior building walls |
| 1 | window | Glass window panes and frames |
| 2 | door | Building entrance/exit doors |
| 3 | balcony | Protruding balcony structures |
| 4 | vegetation | Trees, shrubs, grass |
| 5 | stairs | Exterior stairways |
| 6 | terrain | Ground surface |
| 7 | roof | Roof surfaces and overhangs |
| 8 | blinds | Window blinds/shutters |
| 9 | other | Miscellaneous facade elements |
| 10 | column | Structural columns and pillars |
| 11 | arch | Archways and curved openings |

### HDF5 File Structure

Each `.h5` file contains:
```
point_cloud/xyz        — (N, 3) coordinates, unit-sphere normalised
point_cloud/rgb        — (N, 3) RGB values, [0, 1]
point_cloud/intensity  — (N,)   intensity, [0, 1]
labels/semantic        — (N,)   integer labels 0–11
```

## Benchmark Results

All models trained on 13 buildings, evaluated on 5 held-out test buildings. Results pending.

### Summary

| Model | Epochs | Test mIoU | Test mAcc | Test allAcc |
|-------|--------|-----------|-----------|-------------|
| PTv2 (Point Transformer V2) | 800 | TBD | TBD | TBD |
| PTv3 (Point Transformer V3) | 800 | TBD | TBD | TBD |
| DGCNN | 200 | TBD | TBD | TBD |
| RandLA-Net | 50 | TBD | TBD | TBD |
| PointNet++ (SSG) | 64 | TBD | TBD | TBD |

### Per-Class IoU

| Class | PTv2 | PTv3 | DGCNN | RandLA-Net | PointNet++ |
|-------|------|------|-------|------------|------------|
| wall | TBD | TBD | TBD | TBD | TBD |
| window | TBD | TBD | TBD | TBD | TBD |
| door | TBD | TBD | TBD | TBD | TBD |
| balcony | TBD | TBD | TBD | TBD | TBD |
| vegetation | TBD | TBD | TBD | TBD | TBD |
| stairs | TBD | TBD | TBD | TBD | TBD |
| terrain | TBD | TBD | TBD | TBD | TBD |
| roof | TBD | TBD | TBD | TBD | TBD |
| blinds | TBD | TBD | TBD | TBD | TBD |
| other | TBD | TBD | TBD | TBD | TBD |
| column | TBD | TBD | TBD | TBD | TBD |
| arch | TBD | TBD | TBD | TBD | TBD |
| **mIoU** | TBD | TBD | TBD | TBD | TBD |

## Reproducing Results

All HFX3D-specific integration files (dataset adapters, configs, SLURM scripts) are in the `models/` folder. See `docs/reproduction.md` for step-by-step setup instructions.

| Model | Framework | HFX3D files |
|-------|-----------|-------------|
| PTv2 / PTv3 | [Pointcept](https://github.com/Pointcept/Pointcept) | `models/pointcept/` |
| DGCNN | [AnTao97/dgcnn.pytorch](https://github.com/AnTao97/dgcnn.pytorch) | `models/dgcnn/` |
| RandLA-Net | [QingyongHu/RandLA-Net](https://github.com/QingyongHu/RandLA-Net) | `models/randlanet/` |
| PointNet++ | [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) | `models/pointnet2/` |

## Infrastructure

- **Cluster:** Alliance Canada — `fir.alliancecan.ca`
- **GPUs:** NVIDIA H100 (80 GB)
- **Container:** Apptainer SIF (PyTorch 2.4.1, CUDA 12.1)
- **Scheduler:** SLURM (`sbatch`)

## Repository Structure

```
hfx3d-benchmark/
├── models/
│   ├── dgcnn/
│   │   ├── submit_dgcnn_hfx3d.sh      # SLURM full run (200 epochs, 2×H100)
│   │   └── submit_dgcnn_smoke.sh      # SLURM smoke test (5 epochs)
│   ├── randlanet/
│   │   ├── hfx3d_dataset.py           # HFX3D dataset adapter
│   │   ├── helper_hfx3d.py            # Config and KNN utilities
│   │   ├── submit_randlanet_hfx3d.sh  # SLURM full run (50 epochs, 2×H100)
│   │   └── submit_randlanet_smoke.sh  # SLURM smoke test (3 epochs)
│   ├── pointnet2/
│   │   ├── HFX3DDataLoader.py         # HFX3D dataset adapter
│   │   ├── submit_pointnet2_hfx3d.sh  # SLURM full run (64 epochs, 2×H100)
│   │   └── submit_pointnet2_smoke.sh  # SLURM smoke test (5 epochs)
│   └── pointcept/
│       ├── hfx3d.py                   # HFX3D dataset class (register in Pointcept)
│       ├── semseg-pt-v2m2-0-base.py   # PTv2 config
│       ├── semseg-pt-v3m1-0-base.py   # PTv3 config
│       ├── submit_job_hfx3d.sh        # SLURM PTv2 full run (800 epochs, 2×H100)
│       └── submit_ptv3_hfx3d.sh       # SLURM PTv3 full run (800 epochs, 2×H100)
├── benchmark/results/
│   ├── leaderboard.csv                # mIoU summary table
│   └── baseline_results.json         # per-class IoU for each model
├── configs/
│   └── label_map.json                 # class ID → name mapping
└── docs/
    ├── label_taxonomy.md              # class descriptions and notes
    ├── reproduction.md                # step-by-step reproduction guide
    └── HFX3D_README.md               # building index and naming convention
```
