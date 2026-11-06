# Sourced by the ISBNet SLURM scripts.
ROOT=/scratch/abdiaziz/hfx3d/fresh/isbnet
CODE=$ROOT/ISBNet
SIF=${HFX3D_CONTAINER:-$ROOT/containers/killarney_pointcept.sif}   # ISBNet ops were compiled against this image
TRAIN_NPY=$ROOT/data/hfx3d_isbnet               # converter output (train split, .npy)
H5_ROOT=${HFX3D_DATA_ROOT:-/scratch/abdiaziz/hfx3d/fresh/data/HFX3D_Instance}   # released HFX3D_Instance .h5 (GT + eval input)
RUN="bash $CODE/slurm/run_in_container.sh"      # usage: $RUN python3 tools/train.py ...

stage_train_data () {   # copy the train .npy set to node-local disk
  echo "Copying training data to $SLURM_TMPDIR ($(du -sh $TRAIN_NPY/train | cut -f1)) ..."
  mkdir -p "$SLURM_TMPDIR/hfx3d_isbnet"
  cp -r "$TRAIN_NPY/train" "$SLURM_TMPDIR/hfx3d_isbnet/"
  echo "  done: $(ls $SLURM_TMPDIR/hfx3d_isbnet/train | wc -l) files  $(date)"
}

log_gpu () {   # log_gpu <csv>: sample GPU memory/utilisation every 10 s in the background
  nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
    --format=csv -l 10 > "$1" 2>&1 &
  GPU_LOG_PID=$!
}
