# Reproducing Benchmark Results

All experiments run on Alliance Canada `fir.alliancecan.ca` using SLURM + Apptainer.

## Prerequisites

- Alliance Canada account with `def-jijup` allocation
- Apptainer SIF at `/home/<user>/pointcept.sif` (PyTorch 2.4.1, CUDA 12.1)
- Dataset at `/home/<user>/projects/def-jijup/<user>/WACV_Splits/`

## PTv2m2 and PTv3 (Pointcept)

```bash
git clone https://github.com/Graphical-Spatial-Computing-Lab/hfx3d-pointcept
cd hfx3d-pointcept

# Edit submit_job_hfx3d.sh or submit_ptv3_hfx3d.sh:
#   - Update --bind paths to your home directory

sbatch submit_smoke_test.sh      # verify pipeline (~30 min)
sbatch submit_job_hfx3d.sh       # PTv2 full run (~24h)
sbatch submit_ptv3_hfx3d.sh      # PTv3 full run (~24h)
```

## DGCNN

```bash
git clone https://github.com/Graphical-Spatial-Computing-Lab/hfx3d-dgcnn
cd hfx3d-dgcnn

# Edit submit_dgcnn_hfx3d.sh: update --bind paths and --data_root

sbatch submit_dgcnn_smoke.sh     # verify pipeline (~15 min)
sbatch submit_dgcnn_hfx3d.sh     # full run (~15 min, 200 epochs)
```

## RandLA-Net

```bash
git clone https://github.com/Graphical-Spatial-Computing-Lab/hfx3d-randlanet
cd hfx3d-randlanet

# Edit submit_randlanet_hfx3d.sh: update --bind paths and --data_root

sbatch submit_randlanet_smoke.sh  # verify pipeline (~10 min)
sbatch submit_randlanet_hfx3d.sh  # full run (~24h, 100 epochs)
```

## PointNet++

```bash
git clone https://github.com/Graphical-Spatial-Computing-Lab/hfx3d-pointnet2
cd hfx3d-pointnet2

# Edit submit_pointnet2_hfx3d.sh: update --bind paths and --data_root

sbatch submit_pointnet2_smoke.sh  # verify pipeline (~20 min)
sbatch submit_pointnet2_hfx3d.sh  # full run (~24h, 64 epochs)
```

## Adapting to a New Dataset

Each model repo contains a dataset loader that can be swapped out. See the "Adapting to Your Dataset" section in each repo's README for instructions.
