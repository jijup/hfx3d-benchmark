#!/bin/bash
#SBATCH --job-name=pg_fresh_train
#SBATCH --account=def-jijup_gpu
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --output=/scratch/abdiaziz/hfx3d/fresh/pointgroup/logs/%x-%j.out

# PointGroup training, benchmark protocol:
# from scratch, 18,000 steps (200 epochs x 90 it) at global batch 4 on 4 GPUs, seed 42,
# 5 cm voxels, 7-D input, colour-scale + ignore(-1) fixes, checkpoint every 1,800 steps.
# Checkpoint selection afterwards: full-scene val AP over the 10 saved checkpoints
# (tools/test_fullscene_insseg_hfx3d_fullres.py), then test ONCE with the selected one.
#
# Run:  sbatch /scratch/abdiaziz/hfx3d/fresh/pointgroup/pg_train_tamia.sh

R=/scratch/abdiaziz/hfx3d/fresh/pointgroup/Pointcept
CFG=configs/hfx3d/insseg-pointgroup-v1m1-0-spunet-fresh.py
SIF=/scratch/abdiaziz/hfx3d/containers/pointcept.sif
SEED=${SEED:-42}   # extra seeds: sbatch --export=ALL,SEED=43 ...
EXP=$R/exp/pg_fresh_s$SEED
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
_o=$-; set +u; type module >/dev/null 2>&1 || source /etc/profile.d/z-00-computecanada.sh   # batch shells may lack the module env (nounset off for cvmfs/Lmod)
module load apptainer
case $_o in *u*) set -u;; esac
cd $R
apptainer exec --nv --cleanenv \
  --bind /scratch/abdiaziz/hfx3d:/scratch/abdiaziz/hfx3d --bind /home/abdiaziz:/home/abdiaziz \
  $SIF env PYTHONNOUSERSITE=1 PYTHONPATH=$R WANDB_MODE=disabled \
  python tools/train.py --num-gpus 4 --config-file $CFG --options save_path=$EXP seed=$SEED
rc=$?; echo "train exit=$rc  Finished: $(date)"
ls -la $EXP/model/
exit $rc
