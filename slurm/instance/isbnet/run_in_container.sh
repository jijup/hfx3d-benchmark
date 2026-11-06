#!/bin/bash
# Run "$@" inside the ISBNet container from the repo root (GPU = $CUDA_VISIBLE_DEVICES, default 0).
source /scratch/abdiaziz/hfx3d/fresh/isbnet/ISBNet/slurm/common_isbnet_tamia.sh
exec apptainer exec --nv --cleanenv \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  --env HFX3D_SEED="${HFX3D_SEED:-42}" \
  --bind /scratch/abdiaziz/hfx3d:/scratch/abdiaziz/hfx3d \
  --bind /home/abdiaziz:/home/abdiaziz \
  --bind "${SLURM_TMPDIR:-/tmp}:${SLURM_TMPDIR:-/tmp}" \
  "$SIF" bash -c 'cd "$0" && export PYTHONNOUSERSITE=1 && export PYTHONPATH="$0:$0/isbnet/pointnet2:$0/.deps" && exec "$@"' "$CODE" "$@"
