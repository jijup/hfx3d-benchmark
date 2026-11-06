#!/bin/bash
#SBATCH --job-name=sg_fresh_smoke
#SBATCH --account=def-jijup_gpu
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --output=/scratch/abdiaziz/hfx3d/fresh/softgroup/logs/%x-%j.out
# SoftGroup HFX3D fresh setup -- SMOKE TEST ONLY (50 steps per stage + full-scene
# eval of one val building). Run: sbatch slurm/smoke_tamia.sh  (from the repo root)
ROOT=/scratch/abdiaziz/hfx3d/fresh/softgroup
REPO=$ROOT/SoftGroup
SIF=${HFX3D_CONTAINER:-/scratch/abdiaziz/hfx3d/containers/pointcept.sif}
WD=$ROOT/work_dirs/smoke_$SLURM_JOB_ID
LOG=$ROOT/logs/smoke_$SLURM_JOB_ID
mkdir -p $WD $LOG
_o=$-; set +u; type module >/dev/null 2>&1 || source /etc/profile.d/z-00-computecanada.sh   # batch shells may lack the module env (nounset off for cvmfs/Lmod)
module load apptainer
case $_o in *u*) set -u;; esac
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME start $(date)"

run () {  # $1 = GPU id ("" for none), rest = shell command run inside the container
  local gpu=$1; shift
  apptainer exec --nv --cleanenv \
    --bind /scratch/abdiaziz/hfx3d:/scratch/abdiaziz/hfx3d \
    --bind /home/abdiaziz:/home/abdiaziz \
    $SIF bash -lc "cd $REPO && export PYTHONNOUSERSITE=1 PYTHONPATH=$REPO/.pydeps:$REPO OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$gpu && $*"
}

# host resource monitor
( while true; do echo "== $(date +%T)"; free -g | head -2; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; sleep 30; done ) > $LOG/monitor.txt 2>&1 &
MON=$!

# 0) one-time uncompressed cache of the h5 arrays (train + val BLD008)
T0=$(date +%s)
run "" "python -c \"from softgroup.data.hfx3d import ensure_cache as e; e('$ROOT/data/HFX3D_Instance','train','$ROOT/data/npy_cache',processes=18); e('$ROOT/data/HFX3D_Instance','val','$ROOT/data/npy_cache',names=['HFX_BLD008_ZEB_CLEAN'])\"" > $LOG/cache.log 2>&1
echo "cache built in $(( $(date +%s) - T0 ))s (rc $?)"; du -sh $ROOT/data/npy_cache

# 1) training-split statistics for the grouping config (CPU, background)
run "" "python tools/hfx3d_grouping_stats.py --out $LOG/stats_train.json" > $LOG/stats.log 2>&1 &
STATS=$!

# 2) stage 1 (backbone), 50 steps, ckpt every 25
T0=$(date +%s)
run 0 "python tools/train_hfx3d.py configs/softgroup/softgroup_hfx3d_fresh_backbone.yaml --work_dir $WD/stage1 --stop_after 50 --save_every 25 --num_workers 32" > $LOG/stage1.log 2>&1
echo "stage1 rc $? in $(( $(date +%s) - T0 ))s"

# 3) stage 2 (full), 50 steps from the stage-1 checkpoint
T0=$(date +%s)
run 0 "python tools/train_hfx3d.py configs/softgroup/softgroup_hfx3d_fresh_full.yaml --work_dir $WD/stage2 --pretrain $WD/stage1/step_00050.pth --stop_after 50 --save_every 25 --num_workers 32" > $LOG/stage2.log 2>&1
echo "stage2 rc $? in $(( $(date +%s) - T0 ))s"

# 4) full-scene eval on ONE val building: stage-2 ckpt twice (determinism), stage-1 ckpt once
#    (semantic-only path); plus a resume check of stage 1 on GPU0 (same crops must be replayed)
EV="python tools/test_fullscene_softgroup_hfx3d.py --split val --buildings HFX_BLD008_ZEB_CLEAN"
T0=$(date +%s)
run 1 "timeout 900 $EV --config configs/softgroup/softgroup_hfx3d_fresh_full.yaml --ckpt $WD/stage2/step_05450.pth --out_dir $WD/eval_s2_a" > $LOG/eval_s2_a.log 2>&1 &
E1=$!
run 2 "timeout 900 $EV --config configs/softgroup/softgroup_hfx3d_fresh_full.yaml --ckpt $WD/stage2/step_05450.pth --out_dir $WD/eval_s2_b" > $LOG/eval_s2_b.log 2>&1 &
E2=$!
run 3 "timeout 900 $EV --config configs/softgroup/softgroup_hfx3d_fresh_backbone.yaml --ckpt $WD/stage1/step_00050.pth --out_dir $WD/eval_s1" > $LOG/eval_s1.log 2>&1 &
E3=$!
run 0 "python tools/train_hfx3d.py configs/softgroup/softgroup_hfx3d_fresh_backbone.yaml --work_dir $WD/stage1_resume --resume $WD/stage1/step_00025.pth --stop_after 10 --save_every 1000 --num_workers 32" > $LOG/stage1_resume.log 2>&1
echo "resume check rc $?"
wait $E1; echo "eval_s2_a rc $?"; wait $E2; echo "eval_s2_b rc $?"; wait $E3; echo "eval_s1 rc $?"
echo "evals done in $(( $(date +%s) - T0 ))s"
wait $STATS; echo "stats rc $?"
kill $MON

echo "===== determinism (stage-2 ckpt evaluated twice)"
md5sum $WD/eval_s2_a/*_pred.npz $WD/eval_s2_b/*_pred.npz
grep -h "Full-scene" $WD/eval_s2_a/eval.log $WD/eval_s2_b/eval.log $WD/eval_s1/eval.log
echo "===== resume replay (pts + fingerprint per step must match steps 26-35 of stage1)"
grep -h -o -E "step (2[6-9]|3[0-5])/5400 lr [^ ]* pts [0-9]* fp [0-9.]*" $LOG/stage1.log $LOG/stage1_resume.log
echo "Finished $(date)"
