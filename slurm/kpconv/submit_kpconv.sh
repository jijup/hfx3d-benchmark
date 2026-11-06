#!/bin/bash
#SBATCH --job-name=kpconv_hfx3d_s42
#SBATCH --account=aip-jijup
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --output=logs/%x-%j.out

echo "=== KPConv HFX3D Phase 1 rerun (seed 42, unweighted CE, 7-ch xyz+rgb+intensity) ==="
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Started: $(date)"

mkdir -p /home/a/abdiaziz/KPConv/logs

module load apptainer

WRAPPERS_DIR=/home/a/abdiaziz/KPConv/cpp_wrappers

# Use GPU 0 only (single-GPU model; all 4 allocated per cluster policy)
export CUDA_VISIBLE_DEVICES=0

# Verify C++ extensions are compiled (transferred from Nibi; recompile if missing)
ls ${WRAPPERS_DIR}/cpp_subsampling/*.so 2>/dev/null && echo "cpp_subsampling OK" || {
    echo "Compiling cpp_subsampling..."
    apptainer exec --nv --cleanenv \
        --bind /home/a/abdiaziz:/home/a/abdiaziz \
        ${HFX3D_CONTAINER:-/home/a/abdiaziz/pointcept.sif} bash -lc \
        "cd ${WRAPPERS_DIR}/cpp_subsampling && export PYTHONNOUSERSITE=1 && python setup.py build_ext --inplace"
}
ls ${WRAPPERS_DIR}/cpp_neighbors/*.so 2>/dev/null && echo "cpp_neighbors OK" || {
    echo "Compiling cpp_neighbors..."
    apptainer exec --nv --cleanenv \
        --bind /home/a/abdiaziz:/home/a/abdiaziz \
        ${HFX3D_CONTAINER:-/home/a/abdiaziz/pointcept.sif} bash -lc \
        "cd ${WRAPPERS_DIR}/cpp_neighbors && export PYTHONNOUSERSITE=1 && python setup.py build_ext --inplace"
}

apptainer exec --nv --cleanenv \
  --bind /home/a/abdiaziz:/home/a/abdiaziz \
  --bind /project/6118401/abdiaziz:/project/6118401/abdiaziz \
  ${HFX3D_CONTAINER:-/home/a/abdiaziz/pointcept.sif} bash -lc \
  'cd /home/a/abdiaziz/KPConv && \
   export PYTHONNOUSERSITE=1 && \
   export PYTHONPATH=$PWD && \
   export PYTHONHASHSEED=42 && \
   export CUDA_VISIBLE_DEVICES=0 && \
   python train_HFX3D.py 2>&1'

echo "Finished: $(date)"
