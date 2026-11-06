#!/bin/bash
#SBATCH --job-name=isbnet_fresh_train
#SBATCH --account=def-jijup_gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --output=/scratch/abdiaziz/hfx3d/fresh/isbnet/logs/%x-%j.out

# ISBNet on HFX3D. Seed 42, from scratch, effective batch 4
# (1 H100 x 4 crops), 18,000 optimizer steps in total:
#   stage 1 backbone            6,000 steps  (global    1 -  6,000)  configs/hfx3d/isbnet_backbone_hfx3d.yaml
#   stage 2 instance (frozen bb) 12,000 steps (global 6,001 - 18,000) configs/hfx3d/isbnet_hfx3d.yaml
# Checkpoints every 1,800 GLOBAL steps (+ each stage's last step):
#   work_dirs/backbone_s42/step_{01800,03600,05400,06000}.pth
#   work_dirs/instance_s42/step_{07200,09000,10800,12600,14400,16200,18000}.pth
# Model selection: slurm/eval_isbnet_fresh_tamia.sh on val (highest val AP), then test once.
#
# Resubmit the SAME script to resume: a finished stage is skipped, an unfinished one
# resumes from its latest.pth (model + optimizer + step; data RNG is not restored).
# Run:  sbatch /scratch/abdiaziz/hfx3d/fresh/isbnet/ISBNet/slurm/train_isbnet_fresh_tamia.sh
#
# Only GPU 0 is used (ISBNet's HFX3D loop is single-process); the node is requested
# whole per tamia policy.

source /scratch/abdiaziz/hfx3d/fresh/isbnet/ISBNet/slurm/common_isbnet_tamia.sh
_o=$-; set +u; type module >/dev/null 2>&1 || source /etc/profile.d/z-00-computecanada.sh   # batch shells may lack the module env (nounset off for cvmfs/Lmod)
module load apptainer
case $_o in *u*) set -u;; esac
export CUDA_VISIBLE_DEVICES=0
SEED=${SEED:-42}; export HFX3D_SEED=$SEED   # extra seeds: sbatch --export=ALL,SEED=43 ...
BB=$ROOT/work_dirs/backbone_s$SEED
IN=$ROOT/work_dirs/instance_s$SEED
mkdir -p $BB $IN
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME, start $(date)"
$RUN python3 -c "import torch, spconv, numpy; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'numpy', numpy.__version__, torch.cuda.get_device_name(0))"
log_gpu $ROOT/logs/gpu-$SLURM_JOB_ID.csv
stage_train_data

if [ ! -f $BB/step_06000.pth ]; then
  RES=""; [ -f $BB/latest.pth ] && RES="--resume $BB/latest.pth"
  echo "===== stage 1: backbone, 6,000 steps $RES  $(date)"
  $RUN python3 tools/train.py configs/hfx3d/isbnet_backbone_hfx3d.yaml --only_backbone --skip_validate \
    --data_root $SLURM_TMPDIR/hfx3d_isbnet --work_dir $BB --exp_name backbone_s$SEED $RES || exit 1
else
  echo "stage 1 already finished ($BB/step_06000.pth)"
fi

if [ ! -f $IN/step_18000.pth ]; then
  RES=""; [ -f $IN/latest.pth ] && RES="--resume $IN/latest.pth"
  echo "===== stage 2: instance head, frozen backbone, 12,000 steps $RES  $(date)"
  $RUN python3 tools/train.py configs/hfx3d/isbnet_hfx3d.yaml --skip_validate \
    --data_root $SLURM_TMPDIR/hfx3d_isbnet --work_dir $IN --exp_name instance_s$SEED \
    --pretrain $BB/step_06000.pth $RES || exit 1
else
  echo "stage 2 already finished ($IN/step_18000.pth)"
fi

kill $GPU_LOG_PID 2>/dev/null
md5sum $BB/step_06000.pth $IN/step_*.pth
echo "Finished $(date)"
