#!/bin/bash
#SBATCH --job-name=hfx3d_dgcnn_test
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

echo "Job started: $(date)"
echo "Node: $(hostname)"

mkdir -p ~/DGCNN/logs

module load apptainer

apptainer exec --nv --cleanenv \
  --bind /home/abdiaziz:/home/abdiaziz \
  --bind /home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits:/home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits \
  /home/abdiaziz/pointcept.sif bash -lc \
  'cd ~/DGCNN && \
   export PYTHONNOUSERSITE=1 && \
   export PYTHONPATH="${HOME}/DGCNN:${PYTHONPATH:-}" && \
   python test_semseg_hfx3d.py \
     --exp_name dgcnn_hfx3d_run1 \
     --crop_size 4096 \
     --batch_size 16'

echo "Job finished: $(date)"
