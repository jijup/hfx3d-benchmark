#!/bin/bash
#SBATCH --exclude=fc10608,fc10715
#SBATCH --job-name=randla_hfx3d_s456
#SBATCH --time=3-00:00:00
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
   python main_HFX3D.py \
     --log_dir output_hfx3d_seed456 \
     --max_epoch 100 \
     --batch_size 4 \
     --data_root /home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits \
     --gpus 2 \
     --seed 456'
