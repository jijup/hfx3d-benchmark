#!/bin/bash
# Full DGCNN training on HFX3D — Alliance Canada (fir.alliancecan.ca)
# Update: <user>, <pi>, <dgcnn_dir>, and data_root before submitting.
#SBATCH --job-name=dgcnn_hfx3d
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out

module load apptainer

apptainer exec --nv --cleanenv \
  --bind /home/<user>:/home/<user> \
  --bind /home/<user>/projects/def-<pi>/<user>/WACV_Splits:/home/<user>/projects/def-<pi>/<user>/WACV_Splits \
  /home/<user>/pointcept.sif bash -lc \
  'cd /home/<user>/<dgcnn_dir> && export PYTHONNOUSERSITE=1 && \
   python main_semseg_hfx3d.py \
     --exp_name dgcnn_hfx3d_run1 \
     --data_root /home/<user>/projects/def-<pi>/<user>/WACV_Splits \
     --num_points 4096 \
     --samples_per_building 100 \
     --k 20 \
     --emb_dims 1024 \
     --dropout 0.5 \
     --epochs 200 \
     --batch_size 16 \
     --test_batch_size 16 \
     --lr 1e-3 \
     --scheduler cos'
