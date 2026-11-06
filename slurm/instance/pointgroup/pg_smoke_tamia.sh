#!/bin/bash
#SBATCH --job-name=pg_fresh_smoke
#SBATCH --account=def-jijup_gpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --output=/scratch/abdiaziz/hfx3d/fresh/pointgroup/logs/%x-%j.out

# Smoke test for the PointGroup run:
#   1. numeric checks: colour range through the real train/val pipelines, InstanceParser ignore(-1)
#   2. 2 epochs (180 steps) of training on 4 GPUs
#   3. full-resolution, deterministic full-scene eval of that checkpoint on val -- run TWICE,
#      the two results must be identical

R=/scratch/abdiaziz/hfx3d/fresh/pointgroup/Pointcept
CFG=configs/hfx3d/insseg-pointgroup-v1m1-0-spunet-fresh.py
SIF=${HFX3D_CONTAINER:-/scratch/abdiaziz/hfx3d/containers/pointcept.sif}
DATA=${HFX3D_DATA_ROOT:-/scratch/abdiaziz/hfx3d/fresh/data/HFX3D_Instance}
EXP=$R/exp/pg_fresh_smoke
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
_o=$-; set +u; type module >/dev/null 2>&1 || source /etc/profile.d/z-00-computecanada.sh   # batch shells may lack the module env (nounset off for cvmfs/Lmod)
module load apptainer
case $_o in *u*) set -u;; esac
cd $R
RUN="apptainer exec --nv --cleanenv --bind /scratch/abdiaziz/hfx3d:/scratch/abdiaziz/hfx3d --bind /home/abdiaziz:/home/abdiaziz $SIF env PYTHONNOUSERSITE=1 PYTHONPATH=$R WANDB_MODE=disabled"

echo "=== 1. numeric checks"
for s in train val; do $RUN CHECK_SPLIT=$s python ../pg_check.py 2>&1 | grep -v -i warn | grep -E "InstanceParser|colour|sample"; done

echo "=== 2. training smoke (2 epochs x 90 it)"
$RUN python tools/train.py --num-gpus 4 --config-file $CFG \
  --options save_path=$EXP seed=42 eval_epoch=2 epoch=40 > $EXP.train.log 2>&1
echo "train exit=$?"; grep -E "Train: \[[12]/2\]\[(1|45|90)/90\]|Val result|loss: nan|Error|Traceback" $EXP.train.log | tail -12
ls -la $EXP/model/

echo "=== 3. full-scene val eval (x2, must be identical)"
for k in a b; do
  $RUN python tools/test_fullscene_insseg_hfx3d_fullres.py --config-file $CFG \
    --options save_path=$EXP/eval_val_$k weight=$EXP/model/model_last.pth eval_split=val data.val.data_root=$DATA \
    > $EXP.eval_$k.log 2>&1
  echo "eval $k exit=$?"; grep -E "FULLRES|tiles processed|merged instances|Full-scene test result" $EXP.eval_$k.log
done
diff <(grep -E "AP/AP50/AP25" $EXP.eval_a.log | sed -E 's/^\[[^]]*\] //') <(grep -E "AP/AP50/AP25" $EXP.eval_b.log | sed -E 's/^\[[^]]*\] //') && echo "DETERMINISTIC: identical results"
echo "Finished: $(date)"
