#!/bin/bash
#SBATCH --exclude=fc10608,fc10715
#SBATCH --job-name=dgcnn_hfx3d_s42
#SBATCH --account=def-jijup
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out

module load apptainer

apptainer exec --nv --cleanenv   --bind /home/abdiaziz:/home/abdiaziz   --bind /home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits:/home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits   /home/abdiaziz/pointcept.sif bash -lc   'cd /home/abdiaziz/DGCNN && export PYTHONNOUSERSITE=1 &&    python main_semseg_hfx3d.py      --exp_name dgcnn_hfx3d_seed42      --data_root /home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits      --num_points 4096      --samples_per_building 100      --k 20      --emb_dims 1024      --dropout 0.5      --epochs 200      --batch_size 16      --test_batch_size 16      --lr 1e-3      --use_sgd      --scheduler cos      --seed 42'
