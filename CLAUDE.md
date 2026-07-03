# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HFX3D Benchmark covers three tasks on urban building facade LiDAR point clouds: **semantic segmentation** (available), **instance segmentation**, and **functional-attribute prediction** (both in progress). The dataset covers 26 Halifax, NS buildings scanned with a ZEB Horizon handheld LiDAR. Models are trained and evaluated on a cluster; this repo contains dataset adapters, model configs, SLURM scripts, preprocessing tools, and the annotation pipeline (`scripts/`) — not the training frameworks themselves.

## Infrastructure

- **Cluster:** Alliance Canada — `tamia.alliancecan.ca` (Tamia); allocation account `aip-jijup`
- **Scheduler:** SLURM (`sbatch`) — jobs request a full 4×H100 node (tamia node policy): PTv2/PTv3 use 2 GPUs (`--num-gpus 2`), all other models use 1 (`CUDA_VISIBLE_DEVICES=0`)
- **Container:** Apptainer SIF at `/home/<user>/pointcept.sif` (PyTorch 2.4.1, CUDA 12.1)
- **Dataset on cluster:** `/project/<rap>/<user>/data/HFX3D_Splits/` (e.g. `/project/6118401/abdiaziz/data/HFX3D_Splits/`) with `train/`, `val/`, and `test/` subdirs
- Models are **not cloned into this repo** — files from `models/<framework>/` are copied into separate framework checkouts
- **`HFX3D_DATA_ROOT`** — set this env var to your `HFX3D_Splits/` path before training; all dataset loaders and Pointcept configs read it, falling back to the committed cluster path if unset

## Common Commands

```bash
# Activate the venv first
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate

# Step 1 — strip label-0 points and shift labels (run on raw LAS folder)
python preprocessing/preprocess.py <raw_las_folder> [--output_folder <out>] [--no-backup]

# Step 2 — convert cleaned LAS → HDF5 with official normalization
python preprocessing/las_to_hdf5.py <input_dir> <output_dir>
# input_dir must contain train/ val/ test/ subdirs with .las files

# Analyse class distribution of a LAS directory
python preprocessing/analyse_dataset.py <las_dir> [--output_dir <out>]

# Submit a training job (on cluster, after copying model files)
# All SLURM scripts live in slurm/<framework>/
sbatch slurm/pointcept/submit_ptv2.sh      # PTv2
sbatch slurm/pointcept/submit_ptv3.sh      # PTv3
sbatch slurm/dgcnn/submit_dgcnn.sh         # DGCNN
sbatch slurm/randlanet/submit_randla.sh    # RandLA-Net
sbatch slurm/pointnet2/submit_pn2.sh       # PointNet++
sbatch slurm/pointnet2/submit_pn.sh        # PointNet
sbatch slurm/kpconv/submit_kpconv.sh       # KPConv
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
metadata/num_points    ()      int32     point count
metadata/split         ()      S10       "train" | "val" | "test"
```

**Instance & functional tasks (in progress):** the released splits currently ship `labels/semantic` only. The `scripts/` annotation pipeline adds `labels/instance` (scene-local IDs, `0` = no instance) into a separate working dataset (`Output/{train,val,test}/*.h5`), and temporarily freezes `labels/semantic_original` during class review. Per-instance **functional attributes** (multi-label) are planned and not yet in the repo tooling. See `scripts/README.md`.

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

## Design System & Color Protocol

`configs/palette.py` is the single Python source of truth for all colours. Import from it; never hardcode hex values in scripts.

### 12-class semantic palette

Matches `configs/scalar_fields.xml` (CloudCompare) and all web/JS files exactly.

| ID | Class | Hex |
|----|-------|-----|
| 0 | wall | `#2F4BFF` |
| 1 | window | `#FF7A00` |
| 2 | door | `#00B8D9` |
| 3 | balcony | `#FFD600` |
| 4 | vegetation | `#00C853` |
| 5 | stairs | `#FF1744` |
| 6 | terrain | `#A05A2C` |
| 7 | roof | `#8E24AA` |
| 8 | blinds | `#00E5A8` |
| 9 | other | `#9E9E9E` |
| 10 | column | `#E040FB` |
| 11 | arch | `#9CCC00` |

### Split colours

| Split | Hex |
|-------|-----|
| train | `#2563EB` |
| val | `#F59E0B` |
| test | `#DC2626` |

### Chart rules

- **Class distribution bars/wedges**: each bar/wedge uses the class's canonical hex from `palette.HEX`
- **Per-building heatmaps**: `cmap='Blues'`
- **File statistics bars**: `#2563EB`
- **Style**: `CHART_STYLE` + `CHART_RC` from `configs/palette.py` (whitegrid, no top/right spines)
- **CSS tokens**: `--cls-<name>` and `--split-<train|val|test>` are defined in `static/css/style.css`

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

`label_map.json` is the canonical class-name → ID mapping. `scalar_fields.xml` is the CloudCompare color scale for visualizing the 12 classes. `palette.py` is the single source of truth for all class/split colours. The per-model YAMLs (`dgcnn.yaml`, `pointnet2.yaml`, `pointtransformer.yaml`, `randlanet.yaml`) are **reference summaries** of each model's hyperparameters — the authoritative source is the SLURM script in `slurm/<framework>/` and, for Pointcept, the `models/pointcept/semseg-pt-v*.py` config. Keep the YAML in sync when you change a run.

## Dataset Split

18 train / 4 val / 4 test buildings (26 total). Split definitions live in `data/splits/{train,val,test}.txt` — the analyser reads them at runtime; nothing is hardcoded.

| Split | Buildings |
|-------|-----------|
| Train (18) | BLD001–004, BLD006–007, BLD009–016, BLD018–020, BLD023 |
| Val (4) | BLD008, BLD022, BLD024, BLD025 |
| Test (4) | BLD005, BLD017, BLD021, BLD026 |

## Building File Naming

```
HFX_<BuildingID>_ZEB_CLEAN.las   →   e.g. HFX_BLD002_ZEB_CLEAN.las
```

`BLD001`–`BLD026` (with gaps). Full name-to-ID mapping is in `data/HFX3D/HFX3D_README.md` and `docs/HFX3D_README.md`.
