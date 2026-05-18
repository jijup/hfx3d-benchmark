# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HFX3D Benchmark is a semantic segmentation benchmark for urban building facade LiDAR point clouds. The dataset covers 25 Halifax, NS buildings scanned with a ZEB Horizon handheld LiDAR. Models are trained and evaluated on a cluster; this repo contains dataset adapters, model configs, SLURM scripts, and preprocessing tools — not the training frameworks themselves.

## Infrastructure

- **Cluster:** Alliance Canada — `fir.alliancecan.ca`
- **Scheduler:** SLURM (`sbatch`)
- **Container:** Apptainer SIF at `/home/<user>/pointcept.sif` (PyTorch 2.4.1, CUDA 12.1)
- **Dataset on cluster:** `/home/<user>/projects/def-<pi>/<user>/WACV_Splits/` with `train/` and `test/` subdirs
- Models are **not cloned into this repo** — files from `models/<framework>/` are copied into separate framework checkouts

## Common Commands

```bash
# Activate the venv first
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate

# Step 1 — strip label-0 points and shift labels (run on raw LAS folder)
python preprocessing/preprocess.py <raw_las_folder> [--output_folder <out>] [--no-backup]

# Step 2 — convert cleaned LAS → HDF5 with official normalization
python preprocessing/las_to_hdf5.py <input_dir> <output_dir>
# input_dir must contain train/ and test/ subdirs with .las files

# Analyse class distribution of a LAS directory
python preprocessing/analyse_dataset.py <las_dir> [--output_dir <out>]

# Submit a training job (on cluster, after copying model files)
sbatch submit_job_hfx3d.sh      # PTv2
sbatch submit_ptv3_hfx3d.sh     # PTv3
sbatch submit_dgcnn_hfx3d.sh    # DGCNN
sbatch submit_randlanet_hfx3d.sh
sbatch submit_pointnet2_hfx3d.sh
```

There are no unit tests and no lint configuration in this repo.

## Data Format & Normalization (Official HFX3D Input Protocol)

Every `.h5` file (one per building) must follow this exact structure — all dataset loaders depend on it:

```
point_cloud/xyz        (N, 3)  float32   mean-centered per building, metric scale preserved
point_cloud/rgb        (N, 3)  float32   [0, 1]  (divided by 65535.0)
point_cloud/intensity  (N,)    float32   [0, 1]  global p1/p99 normalization (train-split stats)
labels/semantic        (N,)    uint8     0–11
metadata/centroid      (3,)    float64   original centroid before centering
```

**Normalization rules (locked — apply identically to every model):**
- **XYZ:** `xyz - building_centroid` — metric scale preserved, do NOT unit-sphere normalize
- **RGB:** `rgb / 65535.0` — raw LAS 16-bit → [0, 1]
- **Intensity:** clip to `[p1, p99]` computed from training split, then scale to [0, 1] — preserves cross-building reflectance consistency
- All models receive the same 7D input: `[x, y, z, r, g, b, intensity]`
- Normals, height channels, curvature, and other hand-crafted descriptors are **excluded globally**

`preprocessing/las_to_hdf5.py` implements this. It writes `normalization_stats.json` (intensity p1/p99) alongside the HDF5 files.

## Label Taxonomy (12 classes)

| ID | Class | Frequency note |
|----|-------|---------------|
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

**Class imbalance is severe.** RGB is unreliable — ~70 % of points have near-black values from the ZEB scanner. Models relying heavily on colour underperform. `door`, `column`, and `arch` deviate from the 80/20 split target because their points concentrate in few buildings.

## Architecture

### Dataset adapters (`models/<framework>/`)

Each framework gets its own dataset class that reads the shared HDF5 format:

| Framework | File | Notes |
|-----------|------|-------|
| Pointcept (PTv2/PTv3) | `models/pointcept/hfx3d.py` | Registers `HFX3DDataset` via `@DATASETS.register_module()`. Samples up to 500 k pts/building randomly; falls back to dummy `[0,0,1]` normals if none precomputed. |
| PointNet++ | `models/pointnet2/HFX3DDataLoader.py` | Loads all buildings into RAM, weighted random sampling per `__getitem__`. 9-D feature: centred xyz ‖ rgb ‖ intensity ‖ norm_x ‖ norm_y. Matches S3DIS `in_channel=9+3` convention. |
| RandLA-Net | `models/randlanet/hfx3d_dataset.py` + `helper_hfx3d.py` | C++ KNN replaced with `scipy.cKDTree`. |
| DGCNN | embedded in `main_semseg_hfx3d.py` (copied to DGCNN checkout) | |

Configs in `models/pointcept/semseg-pt-v*.py` set `data_root` — update this to your cluster path before submitting.

### Preprocessing (`preprocessing/`)

Three scripts, run in order:

1. **`preprocess.py`** — strips label-0 (unlabelled) points from raw `.las` files and shifts remaining labels down by 1. Takes `input_folder` as a CLI argument.
2. **`las_to_hdf5.py`** — converts cleaned `.las` → `.h5` with the official normalization. Two-pass: computes intensity p1/p99 from training files first, then converts all splits.
3. **`analyse_dataset.py`** — class-distribution analysis and visualizations. Takes `las_dir` as a CLI argument.

### Configs (`configs/`)

`label_map.json` is the canonical class-name → ID mapping. `scalar_fields.xml` is the CloudCompare color scale for visualizing the 12 classes. YAML files (`default.yaml`, `dgcnn.yaml`, etc.) are placeholders/templates for experiment configuration — contents vary; check before use.

## Dataset Split

20 train / 5 test buildings. Split was chosen by brute-force to minimise per-class deviation from 20 % test target. Test set: `BLD006, BLD014, BLD015, BLD022, BLD024`.

## Building File Naming

```
HFX_<BuildingID>_ZEB_CLEAN.las   →   e.g. HFX_BLD002_ZEB_CLEAN.las
```

`BLD001`–`BLD025` (with gaps). Full name-to-ID mapping is in `data/HFX3D/HFX3D_README.md` and `docs/HFX3D_README.md`.
