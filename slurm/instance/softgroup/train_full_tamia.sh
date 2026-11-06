#!/bin/bash
#SBATCH --job-name=sg_fresh_s42
#SBATCH --account=def-jijup_gpu
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --output=/scratch/abdiaziz/hfx3d/fresh/softgroup/logs/%x-%j.out
# SoftGroup on HFX3D, -- FULL TRAINING (seed 42), 18,000 optimizer
# steps at batch 4: stage 1 backbone 0 -> 5,400, stage 2 full model 5,400 -> 18,000.
# GPU 0 trains. GPUs 1-3 evaluate every checkpoint (every 1,800 global steps) on the
# 4 val buildings with the full-scene pipeline as soon as it is written.
# Idempotent / resumable: resubmit the same script if it hits the time limit; it
# resumes each stage from its latest checkpoint and skips finished evaluations.
#
# HOW TO RUN:  cd /scratch/abdiaziz/hfx3d/fresh/softgroup/SoftGroup && sbatch slurm/train_full_tamia.sh
# Watch:       tail -f /scratch/abdiaziz/hfx3d/fresh/softgroup/logs/sg_fresh_s42-<jobid>.out
# Val curve:   grep -h "Full-scene val" $WD/eval_val/step_*/eval.log
ROOT=/scratch/abdiaziz/hfx3d/fresh/softgroup
REPO=$ROOT/SoftGroup
SIF=${HFX3D_CONTAINER:-/scratch/abdiaziz/hfx3d/containers/pointcept.sif}
SEED=${SEED:-42}   # extra seeds: sbatch --export=ALL,SEED=43 ...
WD=$ROOT/work_dirs/hfx3d_fresh_s$SEED
CFG1=configs/softgroup/softgroup_hfx3d_fresh_backbone.yaml; [ $SEED != 42 ] && CFG1=configs/softgroup/softgroup_hfx3d_fresh_backbone_s$SEED.yaml
CFG2=configs/softgroup/softgroup_hfx3d_fresh_full.yaml; [ $SEED != 42 ] && CFG2=configs/softgroup/softgroup_hfx3d_fresh_full_s$SEED.yaml
mkdir -p $WD/eval_val
_o=$-; set +u; type module >/dev/null 2>&1 || source /etc/profile.d/z-00-computecanada.sh   # batch shells may lack the module env (nounset off for cvmfs/Lmod)
module load apptainer
case $_o in *u*) set -u;; esac
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME start $(date)"

run () {  # $1 = GPU id ("" for none), rest = command inside the container
  local gpu=$1; shift
  apptainer exec --nv --cleanenv \
    --bind /scratch/abdiaziz/hfx3d:/scratch/abdiaziz/hfx3d \
    --bind /home/abdiaziz:/home/abdiaziz \
    $SIF bash -lc "cd $REPO && export PYTHONNOUSERSITE=1 PYTHONPATH=$REPO/.pydeps:$REPO OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$gpu && $*"
}

# uncompressed npy cache of the released h5 (idempotent; built once)
run "" "python -c \"from softgroup.data.hfx3d import ensure_cache as e; [e('$ROOT/data/HFX3D_Instance', s, '$ROOT/data/npy_cache', processes=18) for s in ('train','val')]\""

eval_worker () {  # $1 = GPU, $2 = worker index 0..2 -> evaluates steps k*1800 with k % 3 == $2
  local k=0
  for step in 1800 3600 5400 7200 9000 10800 12600 14400 16200 18000; do
    k=$((k + 1)); [ $(( (k - 1) % 3 )) -ne $2 ] && continue
    local s=$(printf %05d $step); local out=$WD/eval_val/step_$s
    [ -f $out/result.json ] && continue
    while [ ! -f $WD/step_$s.pth ]; do [ -f $WD/TRAINING_ENDED ] && return; sleep 60; done
    local cfg=$CFG2; [ $step -le 5400 ] && cfg=$CFG1
    run $1 "python tools/test_fullscene_softgroup_hfx3d.py --config $cfg --ckpt $WD/step_$s.pth --split val --out_dir $out" > $WD/eval_val/step_$s.log 2>&1
    echo "val eval step $step rc $? $(date)"
  done
}
rm -f $WD/TRAINING_ENDED
eval_worker 1 0 & eval_worker 2 1 & eval_worker 3 2 &

# stage 1: backbone (semantic + offsets), global steps 0 -> 5400
if [ ! -f $WD/step_05400.pth ]; then
  RES=""; [ -L $WD/latest_backbone.pth ] && RES="--resume $WD/latest_backbone.pth"
  run 0 "python tools/train_hfx3d.py $CFG1 --work_dir $WD $RES --num_workers 32"
  echo "stage1 rc $? $(date)"
fi
# stage 2: full model from step_05400.pth (cfg.pretrain), global steps 5400 -> 18000
if [ -f $WD/step_05400.pth ] && [ ! -f $WD/step_18000.pth ]; then
  RES=""; [ -L $WD/latest_full.pth ] && RES="--resume $WD/latest_full.pth"
  run 0 "python tools/train_hfx3d.py $CFG2 --work_dir $WD $RES --num_workers 32"
  echo "stage2 rc $? $(date)"
fi
touch $WD/TRAINING_ENDED
wait
echo "===== val curve (full-scene AP/AP50/AP25; stage-1 steps have no instance head)"
grep -h "Full-scene val" $WD/eval_val/step_*/eval.log
echo "Finished $(date)"
