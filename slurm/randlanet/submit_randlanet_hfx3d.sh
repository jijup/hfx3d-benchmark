#!/bin/bash
# Full RandLA-Net training on HFX3D — Alliance Canada (fir.alliancecan.ca)
# Update: <user>, <pi>, <randlanet_dir>, and data_root before submitting.
#SBATCH --job-name=hfx3d_randlanet
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/%x-%j.out

mkdir -p ~/logs
module load apptainer

apptainer exec --nv --cleanenv \
  --bind /home/<user>:/home/<user> \
  --bind /home/<user>/projects/def-<pi>/<user>/HFX3D_Splits:/home/<user>/projects/def-<pi>/<user>/HFX3D_Splits \
  /home/<user>/pointcept.sif bash -lc \
  'cd /home/<user>/<randlanet_dir> && \
   export PYTHONNOUSERSITE=1 && \
   export OMP_NUM_THREADS=4 && \
   python main_HFX3D.py \
     --log_dir output_hfx3d \
     --max_epoch 100 \
     --batch_size 4 \
     --data_root /home/<user>/projects/def-<pi>/<user>/HFX3D_Splits \
     --gpus 2'
