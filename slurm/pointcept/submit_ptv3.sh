#!/bin/bash
#SBATCH --job-name=ptv3_hfx3d
#SBATCH --account=aip-jijup
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --output=logs/%x-%j.out

# Point Transformer V3 — HFX3D semantic segmentation.
# Cluster: tamia.alliancecan.ca (Alliance Canada). Adjust paths/account for your setup.
module load apptainer

export WANDB_MODE=disabled
export WANDB_DISABLED=true

mkdir -p /home/a/abdiaziz/Pointcept/logs

apptainer exec --nv --cleanenv   --bind /home/a/abdiaziz:/home/a/abdiaziz   --bind /project/6118401/abdiaziz:/project/6118401/abdiaziz   /home/a/abdiaziz/pointcept.sif bash -lc   'cd /home/a/abdiaziz/Pointcept &&    export PYTHONNOUSERSITE=1 &&    export PYTHONPATH=$PWD &&    export WANDB_MODE=disabled &&    export HFX3D_DATA_ROOT="${HFX3D_DATA_ROOT:-/project/6118401/abdiaziz/data/HFX3D_Splits}" &&    python tools/train.py      --num-gpus 2      --config-file $PWD/configs/hfx3d/semseg-pt-v3m1-0-base.py      --options save_path=exp/hfx3d/ptv3_hfx3d seed=42 batch_size=6 enable_wandb=False empty_cache=True'
