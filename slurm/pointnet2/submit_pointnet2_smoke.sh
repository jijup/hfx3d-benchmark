#!/bin/bash
# PointNet++ smoke test (5 epochs) — verify pipeline before full run.
# Update: <user>, <pi>, <pointnet2_dir>, and data_root before submitting.
#SBATCH --job-name=pn2_smoke
#SBATCH --time=0:30:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out

set -euo pipefail
module load apptainer

apptainer exec --nv --cleanenv \
  --bind /home/<user>:/home/<user> \
  --bind /home/<user>/projects/def-<pi>/<user>/HFX3D_Splits:/home/<user>/projects/def-<pi>/<user>/HFX3D_Splits \
  /home/<user>/pointcept.sif bash -lc \
  'cd /home/<user>/<pointnet2_dir> && \
   export PYTHONNOUSERSITE=1 && \
   export PYTHONPATH="${HOME}/<pointnet2_dir>:${PYTHONPATH:-}" && \
   python train_semseg_hfx3d.py \
     --model pointnet2_sem_seg \
     --data_root /home/<user>/projects/def-<pi>/<user>/HFX3D_Splits \
     --epoch 5 \
     --batch_size 8 \
     --npoint 4096 \
     --sample_rate 0.05 \
     --step_size 10 \
     --lr_decay 0.7 \
     --learning_rate 0.001 \
     --log_dir hfx3d_smoke'
