#!/bin/bash
#SBATCH --job-name=ptnet2_hfx3d_s123
#SBATCH --time=7-00:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=/home/abdiaziz/PointNet2/logs/ptnet2_hfx3d_s123-%j.out

mkdir -p ~/PointNet2/logs
module load apptainer

apptainer exec --nv --cleanenv \
  --bind /home/abdiaziz:/home/abdiaziz \
  --bind /home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits:/home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits \
  /home/abdiaziz/pointcept.sif bash -lc \
  'cd ~/PointNet2 && \
   export PYTHONNOUSERSITE=1 && \
   export OMP_NUM_THREADS=4 && \
   python train_semseg_hfx3d.py \
     --model pointnet2_sem_seg \
     --data_root /home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits \
     --epoch 200 \
     --batch_size 16 \
     --npoint 4096 \
     --sample_rate 0.1 \
     --step_size 10 \
     --lr_decay 0.7 \
     --learning_rate 0.001 \
     --log_dir hfx3d_seed123 \
     --seed 123'
