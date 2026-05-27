#!/bin/bash
#SBATCH --job-name=hfx3d_randlanet_eval
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=/home/abdiaziz/RandLA-Net/logs/%x-%j.out

mkdir -p ~/RandLA-Net/logs
module load apptainer

apptainer exec --nv --cleanenv \
  --bind /home/abdiaziz:/home/abdiaziz \
  --bind /home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits:/home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits \
  /home/abdiaziz/pointcept.sif bash -lc \
  'cd ~/RandLA-Net && \
   export PYTHONNOUSERSITE=1 && \
   export OMP_NUM_THREADS=4 && \
   python eval_hfx3d.py \
     --data_root /home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits \
     --checkpoint output_hfx3d_v4/checkpoint_best.tar \
     --batch_size 4'
