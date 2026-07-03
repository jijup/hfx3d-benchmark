#!/bin/bash
# Submit evaluation jobs for HFX3D baselines.
# Run from the repo root: bash slurm/evaluate_all_baselines.sh
# Requires trained checkpoints in each framework's output directory.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Submitting HFX3D baseline evaluation jobs..."

sbatch "$REPO_ROOT/slurm/randlanet/submit_randlanet_eval_fullscene.sh"
echo "  [1/1] RandLA-Net full-scene eval submitted"

echo ""
echo "Note: Pointcept (PTv2/PTv3), DGCNN, and PointNet(++) evaluation"
echo "is run by passing --eval to the training script or using the"
echo "framework's built-in test mode. See docs/REPRODUCTION.md."
echo ""
echo "Monitor with: squeue -u \$USER"
