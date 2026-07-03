#!/bin/bash
#SBATCH --job-name=pn_hfx3d
#SBATCH --account=aip-jijup
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --output=logs/%x-%j.out

# PointNet (pointnet_sem_seg) — HFX3D semantic segmentation.
# Cluster: tamia.alliancecan.ca. Single-GPU model; all 4 GPUs allocated per tamia node policy, GPU 0 used.
module load apptainer
export CUDA_VISIBLE_DEVICES=0

mkdir -p /home/a/abdiaziz/PointNet2/logs

apptainer exec --nv --cleanenv   --bind /home/a/abdiaziz:/home/a/abdiaziz   --bind /project/6118401/abdiaziz:/project/6118401/abdiaziz   /home/a/abdiaziz/pointcept.sif bash -lc   'cd /home/a/abdiaziz/PointNet2 &&    export PYTHONNOUSERSITE=1 &&    export OMP_NUM_THREADS=4 &&    export CUDA_VISIBLE_DEVICES=0 &&    python train_semseg_hfx3d.py      --model pointnet_sem_seg      --data_root "${HFX3D_DATA_ROOT:-/project/6118401/abdiaziz/data/HFX3D_Splits}"      --epoch 200      --batch_size 16      --npoint 4096      --sample_rate 0.1      --step_size 10      --lr_decay 0.7      --learning_rate 0.001      --log_dir pointnet_hfx3d      --seed 42'
