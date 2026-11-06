#!/bin/bash
# Oracle-semantic clustering baselines.
# Submits a 3-stage chain on a CPU cluster (written for fir, account def-jijup):
#   1. derive    : training-split hyperparameters -> cluster_hparams.json
#   2. val sweep : 17-task array (dbscan x5, hdbscan x5, meanshift x5, kmeans x2)
#   3. test      : 4-task array, one per method, run once with its val-selected value
#
# Run:  bash slurm/clustering/run_clustering_fir.sh
# Set ROOT to a directory holding code/ (this repo's models/clustering + evaluation),
# data/HFX3D_Instance/{train,val,test}, and venv/ (python 3.11, numpy, scipy,
# scikit-learn>=1.3, h5py).

ROOT=${ROOT:-/project/def-jijup/abdiaziz/hfx3d_cluster}
ACCOUNT=${ACCOUNT:-def-jijup_cpu}
mkdir -p $ROOT/results $ROOT/logs

COMMON="module load python/3.11 scipy-stack >/dev/null 2>&1; source $ROOT/venv/bin/activate; cd $ROOT/code/models/clustering"
DATA=$ROOT/data/HFX3D_Instance
HP=$ROOT/results/cluster_hparams.json

J1=$(sbatch --parsable --account=$ACCOUNT --job-name=clu_derive --cpus-per-task=16 --mem=96G --time=04:00:00 \
  --output=$ROOT/logs/%x-%j.out --wrap="$COMMON; python derive_cluster_hparams.py --data-root $DATA --out $HP")

# Array index -> (method, grid index)
TASKS="dbscan:0 dbscan:1 dbscan:2 dbscan:3 dbscan:4 hdbscan:0 hdbscan:1 hdbscan:2 hdbscan:3 hdbscan:4 meanshift:0 meanshift:1 meanshift:2 meanshift:3 meanshift:4 kmeans:0 kmeans:1"
J2=$(sbatch --parsable --account=$ACCOUNT --job-name=clu_val --dependency=afterok:$J1 --array=0-16 \
  --cpus-per-task=32 --mem=180G --time=12:00:00 --output=$ROOT/logs/%x-%A_%a.out \
  --wrap="$COMMON; T=(\$(echo $TASKS)); IFS=: read M G <<< \"\${T[\$SLURM_ARRAY_TASK_ID]}\"; \
python eval_cluster_hfx3d.py --method \$M --grid-index \$G --split val --data-root $DATA --hparams $HP --out-dir $ROOT/results")

METHODS="dbscan hdbscan meanshift kmeans"
J3=$(sbatch --parsable --account=$ACCOUNT --job-name=clu_test --dependency=afterany:$J2 --array=0-3 \
  --cpus-per-task=32 --mem=180G --time=16:00:00 --output=$ROOT/logs/%x-%A_%a.out \
  --wrap="$COMMON; M=(\$(echo $METHODS)); M=\${M[\$SLURM_ARRAY_TASK_ID]}; \
G=\$(python select_best.py --method \$M --results-dir $ROOT/results) && \
python eval_cluster_hfx3d.py --method \$M --grid-index \$G --split test --data-root $DATA --hparams $HP --out-dir $ROOT/results --save-pred")

echo "submitted: derive=$J1  val_sweep=$J2  test=$J3"
