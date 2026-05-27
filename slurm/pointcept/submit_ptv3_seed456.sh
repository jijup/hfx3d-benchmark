#!/bin/bash
#SBATCH --job-name=ptv3_hfx3d_s456
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/%x-%j.out

module load apptainer

export WANDB_MODE=disabled
export WANDB_DISABLED=true

mkdir -p /home/abdiaziz/Pointcept/logs

apptainer exec --nv --cleanenv \
  --bind /home/abdiaziz:/home/abdiaziz \
  --bind /home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits:/home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits \
  /home/abdiaziz/pointcept.sif bash -lc \
  'cd /home/abdiaziz/Pointcept && \
   export PYTHONNOUSERSITE=1 && \
   export PYTHONPATH=$PWD && \
   export WANDB_MODE=disabled && \
   python tools/train.py \
     --config-file /home/abdiaziz/Pointcept/configs/hfx3d/semseg-pt-v3m1-0-base.py \
     --options save_path=exp/hfx3d/ptv3_seed456 seed=456 enable_wandb=False empty_cache=True'
