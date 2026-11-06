#!/bin/bash
#SBATCH --job-name=sg_fresh_eval
#SBATCH --account=def-jijup_gpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --output=/scratch/abdiaziz/hfx3d/fresh/softgroup/logs/%x-%j.out
# Full-scene evaluation of ONE SoftGroup checkpoint on a split.
# Protocol: run TEST once, with the checkpoint that has the best val AP.
# HOW TO RUN:
#   cd /scratch/abdiaziz/hfx3d/fresh/softgroup/SoftGroup
#   STEP=12600 SPLIT=test sbatch --export=ALL slurm/eval_tamia.sh
ROOT=/scratch/abdiaziz/hfx3d/fresh/softgroup
REPO=$ROOT/SoftGroup
SIF=${HFX3D_CONTAINER:-/scratch/abdiaziz/hfx3d/containers/pointcept.sif}
SEED=${SEED:-42}   # seeds 43/44: sbatch --export=ALL,SEED=43,STEP=...,SPLIT=test
WD=$ROOT/work_dirs/hfx3d_fresh_s$SEED
: ${STEP:?set STEP (global step of the checkpoint)} ${SPLIT:=test}
S=$(printf %05d $STEP)
SFX=""; [ $SEED != 42 ] && SFX=_s$SEED
CFG=configs/softgroup/softgroup_hfx3d_fresh_full$SFX.yaml; [ $STEP -le 5400 ] && CFG=configs/softgroup/softgroup_hfx3d_fresh_backbone$SFX.yaml
_o=$-; set +u; type module >/dev/null 2>&1 || source /etc/profile.d/z-00-computecanada.sh   # batch shells may lack the module env (nounset off for cvmfs/Lmod)
module load apptainer
case $_o in *u*) set -u;; esac
apptainer exec --nv --cleanenv \
  --bind /scratch/abdiaziz/hfx3d:/scratch/abdiaziz/hfx3d \
  --bind /home/abdiaziz:/home/abdiaziz \
  $SIF bash -lc "cd $REPO && export PYTHONNOUSERSITE=1 PYTHONPATH=$REPO/.pydeps:$REPO OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 && \
  python -c \"from softgroup.data.hfx3d import ensure_cache as e; e('$ROOT/data/HFX3D_Instance','$SPLIT','$ROOT/data/npy_cache')\" && \
  python tools/test_fullscene_softgroup_hfx3d.py --config $CFG --ckpt $WD/step_$S.pth --split $SPLIT --out_dir $WD/eval_${SPLIT}/step_$S"
