# Reproducing Benchmark Results

All experiments run on Alliance Canada `fir.alliancecan.ca` using SLURM + Apptainer.

## Prerequisites

- Alliance Canada account
- Apptainer SIF at `/home/<user>/pointcept.sif` (PyTorch 2.4.1, CUDA 12.1, h5py, scipy, torch_geometric)
  - Build from: https://github.com/Pointcept/Pointcept?tab=readme-ov-file#docker-image
- Dataset at `/home/<user>/projects/def-<pi>/<user>/HFX3D_Splits/` with `train/`, `val/`, and `test/` subdirs

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
cp models/pointcept/semseg-pt-v2m2-smoke.py  /home/<user>/Pointcept/configs/hfx3d/
cp models/pointcept/semseg-pt-v3m1-0-base.py /home/<user>/Pointcept/configs/hfx3d/

# 4. Update data_root in configs to your dataset path

# 5. Copy and update SLURM scripts (replace <user>, <pi>, <pointcept_dir>)
cp slurm/pointcept/submit_job_hfx3d.sh  /home/<user>/Pointcept/
cp slurm/pointcept/submit_ptv3_hfx3d.sh /home/<user>/Pointcept/

# 6. Submit
sbatch submit_job_hfx3d.sh      # PTv2 full run (~24h, 800 epochs)
sbatch submit_ptv3_hfx3d.sh     # PTv3 full run (~24h, 800 epochs)
```

## DGCNN

The HFX3D dataset class is embedded in `main_semseg_hfx3d.py` (no separate file needed).

```bash
# 1. Clone DGCNN
git clone https://github.com/AnTao97/dgcnn.pytorch /home/<user>/DGCNN

# 2. Copy HFX3D training and eval scripts
cp models/dgcnn/main_semseg_hfx3d.py      /home/<user>/DGCNN/
cp models/dgcnn/test_semseg_hfx3d.py      /home/<user>/DGCNN/
cp models/dgcnn/validate_inference_hfx3d.py /home/<user>/DGCNN/

# 3. Copy and update SLURM scripts (replace <user>, <pi>, <dgcnn_dir>)
cp slurm/dgcnn/submit_dgcnn_hfx3d.sh /home/<user>/DGCNN/
cp slurm/dgcnn/submit_dgcnn_smoke.sh  /home/<user>/DGCNN/

# 4. Submit
sbatch submit_dgcnn_smoke.sh    # verify pipeline (~15 min)
sbatch submit_dgcnn_hfx3d.sh   # full run (~15 min, 200 epochs)
```

## RandLA-Net

The C++ KNN extensions from the original repo are replaced with `scipy.cKDTree`.

```bash
# 1. Set up the RandLA-Net framework
#    (use models/randlanet/ files directly — no upstream clone needed for the Python-only version)

# 2. Copy dataset, model, and training files
cp models/randlanet/hfx3d_dataset.py   /home/<user>/RandLA-Net/
cp models/randlanet/helper_hfx3d.py    /home/<user>/RandLA-Net/
cp models/randlanet/RandLANet_HFX3D.py /home/<user>/RandLA-Net/
cp models/randlanet/main_HFX3D.py      /home/<user>/RandLA-Net/
cp models/randlanet/eval_hfx3d.py      /home/<user>/RandLA-Net/
cp models/randlanet/eval_fullscene_hfx3d.py /home/<user>/RandLA-Net/

# 3. Copy and update SLURM scripts (replace <user>, <pi>, <randlanet_dir>)
cp slurm/randlanet/submit_randlanet_hfx3d_v4.sh /home/<user>/RandLA-Net/
cp slurm/randlanet/submit_randlanet_smoke.sh     /home/<user>/RandLA-Net/

# 4. Submit
sbatch submit_randlanet_smoke.sh       # verify pipeline (~10 min)
sbatch submit_randlanet_hfx3d_v4.sh   # full run (~24h, 100 epochs)
```

## PointNet++

```bash
# 1. Clone PointNet++ (PyTorch implementation)
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch /home/<user>/PointNet2

# 2. Copy HFX3D dataset loader and training/eval scripts
cp models/pointnet2/HFX3DDataLoader.py     /home/<user>/PointNet2/data_utils/
cp models/pointnet2/train_semseg_hfx3d.py  /home/<user>/PointNet2/
cp models/pointnet2/test_semseg_hfx3d.py   /home/<user>/PointNet2/

# 3. Copy and update SLURM scripts (replace <user>, <pi>, <pointnet2_dir>)
cp slurm/pointnet2/submit_pointnet2_hfx3d.sh /home/<user>/PointNet2/
cp slurm/pointnet2/submit_pointnet2_smoke.sh  /home/<user>/PointNet2/

# 4. Submit
sbatch submit_pointnet2_smoke.sh    # verify pipeline (~20 min)
sbatch submit_pointnet2_hfx3d.sh   # full run (~24h, 64 epochs)
```

## Adapting to a New Dataset

Each `models/<model>/` folder contains a dataset loader. Replace it with one that:
- Reads from your data format
- Returns the same fields (xyz, features, labels) as the HFX3D version
- Updates `NUM_CLASSES` and class names to match your label set
