# HFX3D Benchmark

Semantic segmentation benchmark for urban building facade LiDAR point clouds. Six baselines evaluated on the HFX3D dataset — 26 buildings in Halifax, NS, scanned with a ZEB Horizon handheld LiDAR.

## Repository Structure

```
hfx3d-benchmark/
├── models/
│   ├── pointcept/       # HFX3DDataset class + PTv2/PTv3 configs
│   ├── dgcnn/           # training, eval, and full-scene inference scripts
│   ├── randlanet/       # dataset adapter, model, training, and eval scripts
│   ├── pointnet2/       # dataset loader + training/eval scripts (PointNet & PointNet++)
│   └── kpconv/          # dataset adapter + training/eval placeholders
├── slurm/
│   ├── pointcept/       # PTv2/PTv3 SLURM scripts (seed 42/123/456)
│   ├── dgcnn/           # DGCNN SLURM scripts (seed 42/123/456)
│   ├── randlanet/       # RandLA-Net SLURM scripts (seed 42/123/456) + eval
│   ├── pointnet2/       # PointNet/PointNet++ SLURM scripts (seed 42/123/456)
│   ├── kpconv/          # KPConv SLURM scripts (seed 42/123/456)
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

Phase 1 baseline: 18 train / 4 val / 4 test, seed 42, 200 epochs, unweighted CE. Full-scene 100%-coverage inference for all models.

### Summary (Phase 1, seed 42)

| Model | OA | mIoU | mP | mR | mF1 |
|-------|-----|------|----|----|-----|
| PointNet++ (SSG) | 78.3% | **28.47%** | 39.1% | 34.3% | 34.8% |
| PointNet | 74.4% | **25.34%** | 39.0% | 31.9% | 32.1% |
| PTv2 (Point Transformer V2) | 78.8% | **19.90%** | 24.1% | 24.7% | 23.6% |
| PTv3 (Point Transformer V3) | 69.5% | **15.79%** | 28.2% | 20.9% | 19.4% |
| DGCNN | 69.3% | **13.04%** | 31.3% | 16.6% | 15.7% |
| RandLA-Net | 41.5% | **5.91%** | 9.9% | 10.8% | 9.0% |
| KPConv | — | — | — | — | — |

### Per-Class IoU (Phase 1, seed 42)

| Class | PTv3 | PTv2 | DGCNN | RandLA-Net | PointNet++ | PointNet | KPConv |
|-------|------|------|-------|------------|------------|---------|--------|
| wall | 73.52% | 75.44% | 61.55% | 41.16% | 72.71% | 67.85% | — |
| window | 11.52% | 16.79% | 4.26% | 6.72% | 21.80% | 16.80% | — |
| door | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.19% | — |
| balcony | 8.87% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | — |
| vegetation | 0.07% | 62.50% | 6.87% | 0.00% | 83.83% | 80.98% | — |
| stairs | 9.02% | 0.00% | 0.00% | 0.00% | 56.58% | 40.50% | — |
| terrain | 85.47% | 84.05% | 83.79% | 22.99% | 82.52% | 72.36% | — |
| roof | 0.00% | 0.00% | 0.00% | 0.00% | 0.04% | 0.43% | — |
| blinds | 0.00% | 0.00% | 0.00% | 0.00% | 8.42% | 2.81% | — |
| other | 0.97% | 0.00% | 0.00% | 0.00% | 15.08% | 20.29% | — |
| column | 0.00% | 0.00% | 0.00% | 0.00% | 0.66% | 1.82% | — |
| arch | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | — |
| **mIoU** | **15.79%** | **19.90%** | **13.04%** | **5.91%** | **28.47%** | **25.34%** | — |

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
sbatch slurm/kpconv/submit_kpconv_seed42.sh    # KPConv     (TODO: not yet implemented)
```

Set `HFX3D_DATA_ROOT` to your `HFX3D_Splits/` path before training; all dataset loaders read this env var.

### Framework Repos

| Model | Upstream |
|-------|----------|
| PTv2 / PTv3 | [Pointcept/Pointcept](https://github.com/Pointcept/Pointcept) |
| DGCNN | [AnTao97/dgcnn.pytorch](https://github.com/AnTao97/dgcnn.pytorch) |
| RandLA-Net | [QingyongHu/RandLA-Net](https://github.com/QingyongHu/RandLA-Net) |
| PointNet / PointNet++ | [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) |
| KPConv | [HuguesTHOMAS/KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch) |

## Preprocessing

The distributed LAS files are already cleaned and labelled (`preprocess.py` was applied by the authors). To produce the HDF5 training data from the distributed LAS files:

```bash
# input_dir must contain train/ val/ test/ subdirs with .las files
python preprocessing/las_to_hdf5.py <input_dir> <output_dir>
```

`las_to_hdf5.py` is two-pass: it computes intensity p1/p99 from training files first, then normalises all splits with those stats. `preprocess.py` is included for reference or if adapting to a new raw dataset.

## Infrastructure

- **Cluster:** Alliance Canada — `fir.alliancecan.ca`
- **GPUs:** NVIDIA H100 (80 GB)
- **Container:** Apptainer SIF (PyTorch 2.4.1, CUDA 12.1)
- **Scheduler:** SLURM
