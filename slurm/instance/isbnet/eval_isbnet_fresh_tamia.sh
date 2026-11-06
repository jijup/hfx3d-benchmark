#!/bin/bash
#SBATCH --job-name=isbnet_fresh_eval
#SBATCH --account=def-jijup_gpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --output=/scratch/abdiaziz/hfx3d/fresh/isbnet/logs/%x-%j.out

# full-scene evaluation of ISBNet instance-stage checkpoints.
# Up to 4 checkpoints run in parallel, one per GPU.
#   Val sweep (model selection, every 1,800 global steps):
#     sbatch slurm/eval_isbnet_fresh_tamia.sh
#   Test, ONCE, with the checkpoint that has the highest val AP (e.g. step 14400):
#     sbatch --export=ALL,SPLIT=test,STEPS=14400 slurm/eval_isbnet_fresh_tamia.sh
# Results: work_dirs/instance_s42/eval_<split>_step_<N>/{result.json,test_fullscene.log}
# (Wall-time note: the old ISBNet eval took ~40 min for the 4 test buildings on one L40S
#  without propagation; the 1-NN propagation adds CPU time. Adjust --time if needed.)

source /scratch/abdiaziz/hfx3d/fresh/isbnet/ISBNet/slurm/common_isbnet_tamia.sh
_o=$-; set +u; type module >/dev/null 2>&1 || source /etc/profile.d/z-00-computecanada.sh   # batch shells may lack the module env (nounset off for cvmfs/Lmod)
module load apptainer
case $_o in *u*) set -u;; esac
SPLIT=${SPLIT:-val}
STEPS=${STEPS:-"07200 09000 10800 12600 14400 16200 18000"}
IN=$ROOT/work_dirs/instance_s${SEED:-42}
echo "Job $SLURM_JOB_ID: split=$SPLIT steps=$STEPS  $(date)"

gpu=0
for st in $STEPS; do
  st=$(printf "%05d" $((10#$st)))
  ck=$IN/step_$st.pth
  [ -f $ck ] || { echo "missing $ck"; continue; }
  out=$IN/eval_${SPLIT}_step_$st
  CUDA_VISIBLE_DEVICES=$gpu $RUN python3 tools/test_fullscene_hfx3d.py configs/hfx3d/isbnet_hfx3d.yaml $ck \
    --data_root $H5_ROOT --eval_split $SPLIT --save_dir $out --no_dump > $ROOT/logs/eval_${SPLIT}_step_$st-$SLURM_JOB_ID.log 2>&1 &
  gpu=$(( (gpu + 1) % 4 ))
  [ $gpu -eq 0 ] && wait
done
wait

for st in $STEPS; do
  st=$(printf "%05d" $((10#$st)))
  grep -h "Full-scene" $IN/eval_${SPLIT}_step_$st/test_fullscene.log 2>/dev/null | sed "s/^/step $st: /"
done
echo "Finished $(date)"
