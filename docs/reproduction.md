# Reproducing Benchmark Results

All experiments run on Alliance Canada `fir.alliancecan.ca` using SLURM + Apptainer.

## Prerequisites

- Alliance Canada account
- Apptainer SIF at `/home/<user>/pointcept.sif` (PyTorch 2.4.1, CUDA 12.1, h5py, scipy, torch_geometric)
  - Build from: https://github.com/Pointcept/Pointcept?tab=readme-ov-file#docker-image
- Dataset at `/home/<user>/projects/def-<pi>/<user>/WACV_Splits/` with `train/` and `test/` subdirs

## PTv2m2 and PTv3 (Pointcept)

```bash
# 1. Clone Pointcept
git clone https://github.com/Pointcept/Pointcept /home/<user>/Pointcept

# 2. Register the HFX3D dataset class
cp models/pointcept/hfx3d.py /home/<user>/Pointcept/pointcept/datasets/hfx3d.py
# Add to /home/<user>/Pointcept/pointcept/datasets/__init__.py:
#   from .hfx3d import HFX3DDataset

# 3. Copy configs
mkdir -p /home/<user>/Pointcept/configs/hfx3d
cp models/pointcept/semseg-pt-v2m2-0-base.py /home/<user>/Pointcept/configs/hfx3d/
cp models/pointcept/semseg-pt-v3m1-0-base.py /home/<user>/Pointcept/configs/hfx3d/

# 4. Update data_root in configs to your dataset path

# 5. Copy and update SLURM scripts (replace <user>, <pi>, <pointcept_dir>)
cp models/pointcept/submit_job_hfx3d.sh  /home/<user>/Pointcept/
cp models/pointcept/submit_ptv3_hfx3d.sh /home/<user>/Pointcept/

# 6. Submit
sbatch submit_job_hfx3d.sh      # PTv2 full run (~24h, 800 epochs)
sbatch submit_ptv3_hfx3d.sh     # PTv3 full run (~24h, 800 epochs)
```

## DGCNN

The HFX3D dataset class is embedded in `main_semseg_hfx3d.py` (no separate file needed).

```bash
# 1. Clone DGCNN
git clone https://github.com/AnTao97/dgcnn.pytorch /home/<user>/DGCNN

# 2. Copy and update SLURM scripts (replace <user>, <pi>, <dgcnn_dir>)
cp models/dgcnn/submit_dgcnn_hfx3d.sh /home/<user>/DGCNN/
cp models/dgcnn/submit_dgcnn_smoke.sh  /home/<user>/DGCNN/

# 3. Submit
sbatch submit_dgcnn_smoke.sh    # verify pipeline (~15 min)
sbatch submit_dgcnn_hfx3d.sh   # full run (~15 min, 200 epochs)
```

## RandLA-Net

The C++ KNN extensions from the original repo are replaced with `scipy.cKDTree`.

```bash
# 1. Set up the RandLA-Net framework
#    (use models/randlanet/ files directly — no upstream clone needed for the Python-only version)

# 2. Copy dataset and config files
cp models/randlanet/hfx3d_dataset.py /home/<user>/RandLA-Net/
cp models/randlanet/helper_hfx3d.py  /home/<user>/RandLA-Net/

# 3. Copy and update SLURM scripts (replace <user>, <pi>, <randlanet_dir>)
cp models/randlanet/submit_randlanet_hfx3d.sh /home/<user>/RandLA-Net/
cp models/randlanet/submit_randlanet_smoke.sh  /home/<user>/RandLA-Net/

# 4. Submit
sbatch submit_randlanet_smoke.sh    # verify pipeline (~10 min)
sbatch submit_randlanet_hfx3d.sh   # full run (~24h, 50 epochs)
```

## PointNet++

```bash
# 1. Clone PointNet++ (PyTorch implementation)
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch /home/<user>/PointNet2

# 2. Add HFX3D dataset loader
cp models/pointnet2/HFX3DDataLoader.py /home/<user>/PointNet2/data_utils/

# 3. Copy and update SLURM scripts (replace <user>, <pi>, <pointnet2_dir>)
cp models/pointnet2/submit_pointnet2_hfx3d.sh /home/<user>/PointNet2/
cp models/pointnet2/submit_pointnet2_smoke.sh  /home/<user>/PointNet2/

# 4. Submit
sbatch submit_pointnet2_smoke.sh    # verify pipeline (~20 min)
sbatch submit_pointnet2_hfx3d.sh   # full run (~24h, 64 epochs)
```

## Adapting to a New Dataset

Each `models/<model>/` folder contains a dataset loader. Replace it with one that:
- Reads from your data format
- Returns the same fields (xyz, features, labels) as the HFX3D version
- Updates `NUM_CLASSES` and class names to match your label set
