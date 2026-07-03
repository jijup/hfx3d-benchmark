#!/bin/bash
#SBATCH --job-name=dgcnn_hfx3d
#SBATCH --account=aip-jijup
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --output=logs/%x-%j.out

# DGCNN — HFX3D semantic segmentation.
# Cluster: tamia.alliancecan.ca. Single-GPU model; all 4 GPUs allocated per tamia node policy, GPU 0 used.
module load apptainer
export CUDA_VISIBLE_DEVICES=0

mkdir -p /home/a/abdiaziz/DGCNN/logs

apptainer exec --nv --cleanenv   --bind /home/a/abdiaziz:/home/a/abdiaziz   --bind /project/6118401/abdiaziz:/project/6118401/abdiaziz   /home/a/abdiaziz/pointcept.sif bash -lc   'cd /home/a/abdiaziz/DGCNN &&    export PYTHONNOUSERSITE=1 &&    export OMP_NUM_THREADS=4 &&    export CUDA_VISIBLE_DEVICES=0 &&    python main_semseg_hfx3d.py      --exp_name dgcnn_hfx3d      --data_root "${HFX3D_DATA_ROOT:-/project/6118401/abdiaziz/data/HFX3D_Splits}"      --num_points 4096      --samples_per_building 100      --k 20      --emb_dims 1024      --dropout 0.5      --epochs 200      --batch_size 16      --test_batch_size 16      --lr 1e-3      --use_sgd      --scheduler cos      --seed 42'
