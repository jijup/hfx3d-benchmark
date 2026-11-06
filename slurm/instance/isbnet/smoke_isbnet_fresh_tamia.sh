#!/bin/bash
#SBATCH --job-name=isbnet_fresh_smoke
#SBATCH --account=def-jijup_gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --output=/scratch/abdiaziz/hfx3d/fresh/isbnet/logs/%x-%j.out

# SMOKE TEST (not a result): short stage 1 -> short stage 2 -> full-scene eval of
# the stage-2 checkpoint on ONE val building (BLD008, the smallest).
# Measures s/step and GPU memory; exercises checkpoint save/load and the eval path.
# Run:  sbatch /scratch/abdiaziz/hfx3d/fresh/isbnet/ISBNet/slurm/smoke_isbnet_fresh_tamia.sh

source /scratch/abdiaziz/hfx3d/fresh/isbnet/ISBNet/slurm/common_isbnet_tamia.sh
_o=$-; set +u; type module >/dev/null 2>&1 || source /etc/profile.d/z-00-computecanada.sh   # batch shells may lack the module env (nounset off for cvmfs/Lmod)
module load apptainer
case $_o in *u*) set -u;; esac
export CUDA_VISIBLE_DEVICES=0
S=$ROOT/smoke/$SLURM_JOB_ID
mkdir -p $S
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME, start $(date)"
log_gpu $S/gpu.csv
stage_train_data

echo "===== stage 1 (backbone) smoke: 150 steps $(date)"
timeout 18m $RUN python3 tools/train.py configs/hfx3d/isbnet_backbone_hfx3d.yaml --only_backbone --skip_validate \
  --data_root $SLURM_TMPDIR/hfx3d_isbnet --work_dir $S/backbone \
  --total_steps 150 --lr_cosine_start_step 60 --save_every 50 --global_step_offset 0
echo "stage 1 exit: $?  $(date)"

echo "===== stage 2 (instance, frozen backbone) smoke: 60 steps $(date)"
timeout 14m $RUN python3 tools/train.py configs/hfx3d/isbnet_hfx3d.yaml --skip_validate \
  --data_root $SLURM_TMPDIR/hfx3d_isbnet --work_dir $S/instance \
  --pretrain $S/backbone/step_00150.pth \
  --total_steps 60 --lr_cosine_start_step 24 --save_every 30 --global_step_offset 150
echo "stage 2 exit: $?  $(date)"

echo "===== full-scene eval, val BLD008 only $(date)"
timeout 18m $RUN python3 tools/test_fullscene_hfx3d.py configs/hfx3d/isbnet_hfx3d.yaml $S/instance/step_00210.pth \
  --data_root $H5_ROOT --eval_split val --only HFX_BLD008_ZEB_CLEAN --save_dir $S/eval_val_BLD008
echo "eval exit: $?  $(date)"

kill $GPU_LOG_PID 2>/dev/null
echo "peak GPU memory (MiB) per GPU:"
awk -F', ' 'NR>1 {split($3,a," "); if (a[1]+0 > m[$2]) m[$2]=a[1]+0} END {for (g in m) print "  gpu " g ": " m[g]}' $S/gpu.csv
ls -la $S/backbone $S/instance $S/eval_val_BLD008
echo "Finished $(date)"
