#!/bin/bash
# Submit seed-42 training jobs for all HFX3D baselines.
# Run from the repo root: bash slurm/run_all_baselines.sh
# For multi-seed runs, submit the seed123/seed456 variants individually.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Submitting HFX3D baseline training jobs (seed 42)..."

sbatch "$REPO_ROOT/slurm/pointcept/submit_ptv2_seed42.sh"
echo "  [1/7] PTv2        submitted"

sbatch "$REPO_ROOT/slurm/pointcept/submit_ptv3_seed42.sh"
echo "  [2/7] PTv3        submitted"

sbatch "$REPO_ROOT/slurm/dgcnn/submit_dgcnn_seed42.sh"
echo "  [3/7] DGCNN       submitted"

sbatch "$REPO_ROOT/slurm/randlanet/submit_randla_seed42.sh"
echo "  [4/7] RandLA-Net  submitted"

sbatch "$REPO_ROOT/slurm/pointnet2/submit_pn2_seed42.sh"
echo "  [5/7] PointNet++  submitted"

sbatch "$REPO_ROOT/slurm/pointnet2/submit_pn_seed42.sh"
echo "  [6/7] PointNet    submitted"

sbatch "$REPO_ROOT/slurm/kpconv/submit_kpconv_seed42.sh"
echo "  [7/7] KPConv      submitted"

echo "Done. Monitor with: squeue -u \$USER"
