#!/bin/bash
#SBATCH --job-name=hfx3d_smoke
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
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
     --config-file /home/abdiaziz/Pointcept/configs/hfx3d/semseg-pt-v2m2-smoke.py \
     --options save_path=exp/hfx3d/smoke_test enable_wandb=False'
