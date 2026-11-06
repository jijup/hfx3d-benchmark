#!/bin/bash
#SBATCH --job-name=pg_fresh_select_test
#SBATCH --account=def-jijup_gpu
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --output=/scratch/abdiaziz/hfx3d/fresh/pointgroup/logs/%x-%j.out

# checkpoint selection for the PointGroup run:
#   1. full-scene VAL eval (tools/test_fullscene_insseg_hfx3d_fullres.py) of every saved
#      checkpoint (epoch_20 ... epoch_200, i.e. every 1,800 steps), 4 GPUs in parallel;
#   2. pick the highest val AP (ties: AP50, then the earlier checkpoint);
#   3. evaluate the TEST split ONCE with that checkpoint.
#
# Run (after pg_fresh_train finishes):
#   sbatch --dependency=afterok:<train_job> /scratch/abdiaziz/hfx3d/fresh/pointgroup/pg_select_test_tamia.sh

R=/scratch/abdiaziz/hfx3d/fresh/pointgroup/Pointcept
CFG=configs/hfx3d/insseg-pointgroup-v1m1-0-spunet-fresh.py
SIF=/scratch/abdiaziz/hfx3d/containers/pointcept.sif
DATA=/scratch/abdiaziz/hfx3d/fresh/data/HFX3D_Instance
EXP=$R/exp/pg_fresh_s${SEED:-42}
OUT=$EXP/selection
mkdir -p $OUT
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
_o=$-; set +u; type module >/dev/null 2>&1 || source /etc/profile.d/z-00-computecanada.sh   # batch shells may lack the module env (nounset off for cvmfs/Lmod)
module load apptainer
case $_o in *u*) set -u;; esac
cd $R
run () {  # $1 GPU, $2 checkpoint file, $3 split, $4 tag
  CUDA_VISIBLE_DEVICES=$1 apptainer exec --nv --cleanenv \
    --bind /scratch/abdiaziz/hfx3d:/scratch/abdiaziz/hfx3d --bind /home/abdiaziz:/home/abdiaziz \
    $SIF env PYTHONNOUSERSITE=1 PYTHONPATH=$R \
    python tools/test_fullscene_insseg_hfx3d_fullres.py --config-file $CFG \
      --options save_path=$OUT/$4 weight=$EXP/model/$2 eval_split=$3 data.val.data_root=$DATA \
    > $OUT/$4.log 2>&1
}

# 1. val sweep, 4 checkpoints at a time (one per GPU)
CKPTS=$(cd $EXP/model && ls epoch_*.pth | sort -t_ -k2 -n)
echo "checkpoints: $CKPTS"
i=0
for c in $CKPTS; do
  run $((i % 4)) $c val val_${c%.pth} &
  i=$((i + 1))
  if [ $((i % 4)) -eq 0 ]; then wait; fi
done
wait

# 2. select on val AP
echo "ckpt mAP AP50 AP25" > $OUT/val_sweep.txt
for c in $CKPTS; do
  t=val_${c%.pth}
  r=$(grep -o -E "Full-scene val result: mAP/AP50/AP25 [0-9.]+/[0-9.]+/[0-9.]+" $OUT/$t.log | awk '{print $NF}' | tr '/' ' ')
  echo "$c $r" >> $OUT/val_sweep.txt
done
cat $OUT/val_sweep.txt
BEST=$(tail -n +2 $OUT/val_sweep.txt | awk 'NF==4' | sort -k2,2gr -k3,3gr -s | head -1 | awk '{print $1}')
echo "SELECTED (best val AP): $BEST" | tee $OUT/selected.txt

# 3. test once
run 0 $BEST test test_${BEST%.pth}
grep -E "Full-scene test result|AP/AP50/AP25" $OUT/test_${BEST%.pth}.log | sed -E 's/^\[[^]]*\] //'
echo "Finished: $(date)"
