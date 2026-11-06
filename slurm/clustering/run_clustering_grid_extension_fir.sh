#!/bin/bash
# Grid-edge extension for the clustering baselines.
# Round 1 (run_clustering_fir.sh) put the val optimum on the grid edge for DBSCAN,
# MeanShift and KMeans. This chain:
#   1. re-derives the training-split hyperparameters with the extra statistics
#      (MeanShift q=0.5, instance-size p75) into cluster_hparams_v12.json and checks
#      every value round 1 used is unchanged;
#   2. runs val for the new grid points only (dbscan g5-g7, meanshift g5, kmeans g2);
#   3. archives round-1 test outputs, re-selects on the full val sweep and runs test once.
#
# Run:  bash slurm/clustering/run_clustering_grid_extension_fir.sh

ROOT=${ROOT:-/project/def-jijup/abdiaziz/hfx3d_cluster}
ACCOUNT=${ACCOUNT:-def-jijup_cpu}
COMMON="module load python/3.11 scipy-stack >/dev/null 2>&1; source $ROOT/venv/bin/activate; cd $ROOT/code/models/clustering"
DATA=$ROOT/data/HFX3D_Instance
HP1=$ROOT/results/cluster_hparams.json
HP=$ROOT/results/cluster_hparams_v12.json

CHECK="import json; a=json.load(open('$HP1')); b=json.load(open('$HP')); \
bad=[(c,k) for c in a['classes'] for k,v in a['classes'][c].items() if k!='meanshift_bandwidth' and b['classes'][c][k]!=v] + \
[(c,q) for c in a['classes'] for q,v in a['classes'][c]['meanshift_bandwidth'].items() if b['classes'][c]['meanshift_bandwidth'][q]!=v]; \
print('round-1 hyperparameters unchanged' if not bad else 'CHANGED: %s' % bad); assert not bad"

J1=$(sbatch --parsable --account=$ACCOUNT --job-name=clu_derive2 --cpus-per-task=16 --mem=96G --time=02:00:00 \
  --output=$ROOT/logs/%x-%j.out --wrap="$COMMON; python derive_cluster_hparams.py --data-root $DATA --out $HP && python -c \"$CHECK\"")

TASKS="dbscan:5 dbscan:6 dbscan:7 meanshift:5 kmeans:2"
J2=$(sbatch --parsable --account=$ACCOUNT --job-name=clu_val2 --dependency=afterok:$J1 --array=0-4 \
  --cpus-per-task=32 --mem=180G --time=06:00:00 --output=$ROOT/logs/%x-%A_%a.out \
  --wrap="$COMMON; T=(\$(echo $TASKS)); IFS=: read M G <<< \"\${T[\$SLURM_ARRAY_TASK_ID]}\"; \
python eval_cluster_hfx3d.py --method \$M --grid-index \$G --split val --data-root $DATA --hparams $HP --out-dir $ROOT/results")

METHODS="dbscan meanshift kmeans"
# PREV_TEST: round-1 test array still running, if any (its outputs must exist before archiving)
J3=$(sbatch --parsable --account=$ACCOUNT --job-name=clu_test2 --dependency=afterany:$J2${PREV_TEST:+:$PREV_TEST} --array=0-2 \
  --cpus-per-task=32 --mem=180G --time=08:00:00 --output=$ROOT/logs/%x-%A_%a.out \
  --wrap="$COMMON; M=(\$(echo $METHODS)); M=\${M[\$SLURM_ARRAY_TASK_ID]}; \
mkdir -p $ROOT/results/round1_edge_selection; mv $ROOT/results/\${M}_g*_test* $ROOT/results/\${M}_selection.json $ROOT/results/round1_edge_selection/ 2>/dev/null; \
G=\$(python select_best.py --method \$M --results-dir $ROOT/results) && \
python eval_cluster_hfx3d.py --method \$M --grid-index \$G --split test --data-root $DATA --hparams $HP --out-dir $ROOT/results --save-pred")

echo "submitted: derive2=$J1  val2=$J2  test2=$J3"
