#!/bin/bash
# Submit seed-42 training jobs for all HFX3D baselines.
# Run from the repo root: bash slurm/run_all_baselines.sh
# For multi-seed runs, submit the seed123/seed456 variants individually.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Submitting HFX3D baseline training jobs (seed 42)..."

sbatch "$REPO_ROOT/slurm/pointcept/submit_ptv2_seed42.sh"
echo "  [1/6] PTv2        submitted"

sbatch "$REPO_ROOT/slurm/pointcept/submit_ptv3_seed42.sh"
echo "  [2/6] PTv3        submitted"

sbatch "$REPO_ROOT/slurm/dgcnn/submit_dgcnn_seed42.sh"
echo "  [3/6] DGCNN       submitted"

sbatch "$REPO_ROOT/slurm/randlanet/submit_randla_seed42.sh"
echo "  [4/6] RandLA-Net  submitted"

sbatch "$REPO_ROOT/slurm/pointnet2/submit_pn2_seed42.sh"
echo "  [5/6] PointNet++  submitted"

sbatch "$REPO_ROOT/slurm/pointnet2/submit_pn_seed42.sh"
echo "  [6/6] PointNet    submitted"

echo "Done. Monitor with: squeue -u \$USER"
