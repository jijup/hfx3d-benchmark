#!/bin/bash
#SBATCH --job-name=pointcept_hfx3d
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=logs/%x-%j.out

###############################
# CONFIGURE YOUR PATHS HERE
###############################

# Where your home is
HOME_DIR=/home/abdiaziz

# Where to store/locate the SIF
SIF_PATH=$HOME_DIR/pointcept.sif

# Repo location
REPO_DIR=$HOME_DIR/Pointcept

# Dataset dirs
HFX3D_DIR=$HOME_DIR/hfx3d_processed
WACV_SPLITS_DIR=$HOME_DIR/projects/def-jijup/abdiaziz/WACV_Splits

# Config file
CONFIG_FILE=$REPO_DIR/configs/hfx3d/semseg-pt-v2m2-0-base.py

###############################
# BEGIN JOB
###############################

echo "=== Loading modules ==="
module purge
module load apptainer

echo "=== Disabling WandB ==="
export WANDB_MODE=disabled
export WANDB_DISABLED=true

############################################
# Step 1: Pull the Apptainer image if missing
############################################
if [ ! -f "$SIF_PATH" ]; then
    echo "=== SIF not found — pulling Apptainer image ==="
    apptainer pull $SIF_PATH docker://pointcept/pointcept:v1.6.0-pytorch2.5.0-cuda12.4-cudnn9-devel
else
    echo "=== Using existing Apptainer image: $SIF_PATH ==="
fi

############################################
# Step 2: Clone Repo if Missing
############################################
if [ ! -d "$REPO_DIR" ]; then
    echo "=== Cloning Pointcept repo ==="
    git clone https://github.com/Pointcept/Pointcept.git $REPO_DIR
else
    echo "=== Repo exists — pulling latest changes ==="
    cd $REPO_DIR && git pull
fi

############################################
# Step 3: Run Training
############################################
echo "=== Starting training in Apptainer ==="

apptainer exec --nv --cleanenv \
  --bind $HOME_DIR:$HOME_DIR \
  --bind $HFX3D_DIR:$HFX3D_DIR \
  --bind $WACV_SPLITS_DIR:$WACV_SPLITS_DIR \
  $SIF_PATH bash -lc "
    cd $REPO_DIR && \
    export PYTHONNOUSERSITE=1 && \
    export PYTHONPATH=$REPO_DIR && \
    export WANDB_MODE=disabled && \
    python tools/train.py \
      --config-file $CONFIG_FILE \
      --options save_path=exp/hfx3d/run1 enable_wandb=False
"

echo "=== JOB COMPLETE ==="