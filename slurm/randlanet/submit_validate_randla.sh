#!/bin/bash
#SBATCH --job-name=randla_validate
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/abdiaziz/RandLA-Net/logs/randla_validate-%j.out

mkdir -p ~/RandLA-Net/logs
module load apptainer

apptainer exec --nv --cleanenv \
  --bind /home/abdiaziz:/home/abdiaziz \
  --bind /home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits:/home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits \
  /home/abdiaziz/pointcept.sif bash -lc \
  'cd ~/RandLA-Net && \
   export PYTHONNOUSERSITE=1 && \
   export OMP_NUM_THREADS=4 && \
   python validate_eval_hfx3d.py'
