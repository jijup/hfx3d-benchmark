#!/bin/bash
# DGCNN smoke test (5 epochs) — verify pipeline before full run.
# Update: <user>, <pi>, <dgcnn_dir>, and data_root before submitting.
#SBATCH --job-name=dgcnn_smoke
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out

module load apptainer

apptainer exec --nv --cleanenv \
  --bind /home/<user>:/home/<user> \
  --bind /home/<user>/projects/def-<pi>/<user>/HFX3D_Splits:/home/<user>/projects/def-<pi>/<user>/HFX3D_Splits \
  /home/<user>/pointcept.sif bash -lc \
  'cd /home/<user>/<dgcnn_dir> && export PYTHONNOUSERSITE=1 && \
   python main_semseg_hfx3d.py \
     --exp_name dgcnn_smoke \
     --data_root /home/<user>/projects/def-<pi>/<user>/HFX3D_Splits \
     --num_points 4096 \
     --samples_per_building 10 \
     --k 20 \
     --emb_dims 1024 \
     --dropout 0.5 \
     --epochs 5 \
     --batch_size 8 \
     --test_batch_size 8 \
     --lr 1e-3 \
     --scheduler cos'
