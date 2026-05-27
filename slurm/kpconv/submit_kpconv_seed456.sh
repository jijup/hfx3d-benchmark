#!/bin/bash
#SBATCH --exclude=fc10608,fc10715
#SBATCH --job-name=kpconv_hfx3d_s456
#SBATCH --account=def-jijup
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=/home/abdiaziz/KPConv/logs/kpconv_hfx3d_s456-%j.out

mkdir -p ~/KPConv/logs
module load apptainer

# TODO: update --data_root and paths once train_hfx3d.py is implemented
apptainer exec --nv --cleanenv \
  --bind /home/abdiaziz:/home/abdiaziz \
  --bind /home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits:/home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits \
  /home/abdiaziz/pointcept.sif bash -lc \
  'cd ~/KPConv && \
   export PYTHONNOUSERSITE=1 && \
   export OMP_NUM_THREADS=4 && \
   python train_hfx3d.py \
     --data_root /home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits \
     --log_dir output_hfx3d_seed456 \
     --seed 456'
