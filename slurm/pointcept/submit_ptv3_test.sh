#!/bin/bash
#SBATCH --job-name=hfx3d_ptv3_test
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --output=/home/abdiaziz/Pointcept/logs/%x-%j.out

module load apptainer
mkdir -p /home/abdiaziz/Pointcept/logs

apptainer exec --nv --cleanenv \
  --bind /home/abdiaziz:/home/abdiaziz \
  --bind /home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits:/home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits \
  /home/abdiaziz/pointcept.sif bash -lc \
  'cd /home/abdiaziz/Pointcept && \
   export PYTHONNOUSERSITE=1 && \
   export PYTHONPATH=$PWD && \
   export WANDB_MODE=disabled && \
   rm -f exp/hfx3d/ptv3_run1/result/*.npy && \
   python tools/test.py \
     --config-file configs/hfx3d/semseg-pt-v3m1-0-base.py \
     --num-gpus 1 \
     --options save_path=exp/hfx3d/ptv3_run1 weight=exp/hfx3d/ptv3_run1/model/model_best.pth batch_size=2 num_worker=0 empty_cache=True'
