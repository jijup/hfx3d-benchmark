#!/bin/bash
# Full PointNet++ training on HFX3D — Alliance Canada (fir.alliancecan.ca)
# Update: <user>, <pi>, <pointnet2_dir>, and data_root before submitting.
#SBATCH --job-name=hfx3d_pointnet2
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p ~/logs
module load apptainer

apptainer exec --nv --cleanenv \
  --bind /home/<user>:/home/<user> \
  --bind /home/<user>/projects/def-<pi>/<user>/WACV_Splits:/home/<user>/projects/def-<pi>/<user>/WACV_Splits \
  /home/<user>/pointcept.sif bash -lc \
  'cd /home/<user>/<pointnet2_dir> && \
   export PYTHONNOUSERSITE=1 && \
   export PYTHONPATH="${HOME}/<pointnet2_dir>:${PYTHONPATH:-}" && \
   python train_semseg_hfx3d.py \
     --model pointnet2_sem_seg \
     --data_root /home/<user>/projects/def-<pi>/<user>/WACV_Splits \
     --epoch 64 \
     --batch_size 16 \
     --npoint 4096 \
     --step_size 10 \
     --lr_decay 0.7 \
     --learning_rate 0.001 \
     --log_dir hfx3d_run1'
