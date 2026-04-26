#!/bin/bash
# PTv2 full training on HFX3D — Alliance Canada (fir.alliancecan.ca)
# Update: <user>, <pi>, <pointcept_dir>, and data_root in the config before submitting.
#SBATCH --job-name=hfx3d_ptv2
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=logs/%x-%j.out

module load apptainer
export WANDB_MODE=disabled
mkdir -p /home/<user>/<pointcept_dir>/logs

apptainer exec --nv --cleanenv \
  --bind /home/<user>:/home/<user> \
  --bind /home/<user>/projects/def-<pi>/<user>/WACV_Splits:/home/<user>/projects/def-<pi>/<user>/WACV_Splits \
  /home/<user>/pointcept.sif bash -lc \
  'cd /home/<user>/<pointcept_dir> && \
   export PYTHONNOUSERSITE=1 && \
   export PYTHONPATH=$PWD && \
   export WANDB_MODE=disabled && \
   python tools/train.py \
     --config-file configs/hfx3d/semseg-pt-v2m2-0-base.py \
     --options save_path=exp/hfx3d/ptv2_run1 enable_wandb=False'
