#!/bin/bash
# RandLA-Net smoke test (3 epochs) — verify pipeline before full run.
# Update: <user>, <pi>, <randlanet_dir>, and data_root before submitting.
#SBATCH --job-name=randlanet_smoke
#SBATCH --time=0:30:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=%x-%j.out

module load apptainer

apptainer exec --nv --cleanenv \
  --bind /home/<user>:/home/<user> \
  --bind /home/<user>/projects/def-<pi>/<user>/WACV_Splits:/home/<user>/projects/def-<pi>/<user>/WACV_Splits \
  /home/<user>/pointcept.sif bash -lc \
  'cd /home/<user>/<randlanet_dir> && \
   export PYTHONNOUSERSITE=1 && \
   export OMP_NUM_THREADS=4 && \
   python main_HFX3D.py \
     --log_dir output_smoke \
     --max_epoch 3 \
     --batch_size 4 \
     --data_root /home/<user>/projects/def-<pi>/<user>/WACV_Splits \
     --gpus 1'
