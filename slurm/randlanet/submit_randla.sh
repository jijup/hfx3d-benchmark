#!/bin/bash
#SBATCH --job-name=randla_hfx3d
#SBATCH --account=aip-jijup
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --output=logs/%x-%j.out

# RandLA-Net — HFX3D semantic segmentation.
# Cluster: tamia.alliancecan.ca. Single-GPU model; all 4 GPUs allocated per tamia node policy, GPU 0 used.
module load apptainer
export CUDA_VISIBLE_DEVICES=0

mkdir -p /home/a/abdiaziz/RandLA-Net/logs

apptainer exec --nv --cleanenv   --bind /home/a/abdiaziz:/home/a/abdiaziz   --bind /project/6118401/abdiaziz:/project/6118401/abdiaziz   /home/a/abdiaziz/pointcept.sif bash -lc   'cd /home/a/abdiaziz/RandLA-Net &&    export PYTHONNOUSERSITE=1 &&    export OMP_NUM_THREADS=4 &&    export CUDA_VISIBLE_DEVICES=0 &&    python main_HFX3D.py      --log_dir output_hfx3d      --max_epoch 200      --batch_size 4      --data_root "${HFX3D_DATA_ROOT:-/project/6118401/abdiaziz/data/HFX3D_Splits}"      --gpus 1      --seed 42'
