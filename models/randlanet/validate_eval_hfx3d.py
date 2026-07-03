#!/usr/bin/env python3
"""
Smoke-test eval_fullscene_hfx3d.py pipeline on HFX_BLD015 (smallest test building, ~1.9M pts).

Uses randomly-initialised weights — correctness of predictions is not the goal;
correctness of the pipeline mechanics is.

Checks
------
  1. Coverage: vote_count.min() >= 1 after the loop (100% of points predicted)
  2. Prediction count equals original point count N
  3. Confusion matrix rows/cols == NUM_CLASSES, matrix sums to N, every GT class
     row sums to its true count
  4. No NaNs or Infs anywhere in prob_sum or averaged probabilities; avg_prob
     row sums ≈ 1.0
  5. VmRSS (current RSS via /proc/self/status) does not grow by more than 500 MB
     between the start of the building and after deleting all per-building arrays
"""
import os, sys, time, gc
import numpy as np
import h5py
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.expanduser('~/RandLA-Net'))
from helper_hfx3d import ConfigHFX3D as cfg, DataProcessing as DP
from hfx3d_dataset import NUM_CLASSES, LABEL_TO_NAMES
from RandLANet_HFX3D import Network, compute_loss

CLASS_NAMES = [LABEL_TO_NAMES[i] for i in range(NUM_CLASSES)]
N_PTS     = cfg.num_points        # 45056
MIN_VOTES = 1
BUILDING  = '/project/6118401/abdiaziz/data/HFX3D_Splits/test/HFX_BLD015_ZEB_CLEAN.h5'

PASS = '✓ PASS'
FAIL = '✗ FAIL'


def log(msg):
    print(msg, flush=True)


def current_rss_mb():
    """Current resident set size in MB, from /proc/self/status (Linux)."""
    try:
        with open('/proc/self/status') as fh:
            for line in fh:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0   # kB -> MB
    except Exception:
        pass
    return float('nan')


# ---------------------------------------------------------------------------
# Model (random weights — no checkpoint)
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Device: {device}")

net = Network(cfg)
net.to(device)
net.eval()
log("Model: randomly-initialised weights (pipeline validation, not accuracy)")


# ---------------------------------------------------------------------------
# build_inputs — identical to eval_fullscene_hfx3d.py
# ---------------------------------------------------------------------------
def build_inputs(sel_xyz, sel_feats, sel_idx):
    batch_xyz   = sel_xyz[np.newaxis]
    batch_feats = sel_feats[np.newaxis]
    batch_label = np.zeros((1, N_PTS), dtype=np.int32)
    batch_idx   = sel_idx[np.newaxis]
    batch_cind  = np.zeros((1, 1), dtype=np.int32)

    input_points, input_neighbors, input_pools, input_up_samples = [], [], [], []
    cur = batch_xyz.copy()
    for i in range(cfg.num_layers):
        neigh = DP.knn_search(cur, cur, cfg.k_n)
        n_sub = cur.shape[1] // cfg.sub_sampling_ratio[i]
        sub   = cur[:, :n_sub, :]
        pool  = neigh[:, :n_sub, :]
        up    = DP.knn_search(sub, cur, 1)
        input_points.append(cur)
        input_neighbors.append(neigh)
        input_pools.append(pool)
        input_up_samples.append(up)
        cur = sub

    return {
        'xyz':        [torch.from_numpy(p).float().to(device) for p in input_points],
        'neigh_idx':  [torch.from_numpy(n).long().to(device)  for n in input_neighbors],
        'sub_idx':    [torch.from_numpy(p).long().to(device)  for p in input_pools],
        'interp_idx': [torch.from_numpy(u).long().to(device)  for u in input_up_samples],
        'features':   torch.from_numpy(batch_feats).float().permute(0, 2, 1).to(device),
        'labels':     torch.from_numpy(batch_label).long().to(device),
        'input_inds': torch.from_numpy(batch_idx).long().to(device),
        'cloud_inds': torch.from_numpy(batch_cind).long().to(device),
    }


# ---------------------------------------------------------------------------
# Load building
# ---------------------------------------------------------------------------
log(f"\nBuilding: {os.path.basename(BUILDING)}")
with h5py.File(BUILDING, 'r') as f:
    xyz    = f['point_cloud/xyz'][:].astype(np.float32)
    rgb    = f['point_cloud/rgb'][:]
    inten  = f['point_cloud/intensity'][:]
    labels = f['labels/semantic'][:].astype(np.int64)

feats = np.concatenate([xyz, rgb, inten[:, np.newaxis]], axis=1).astype(np.float32)
tree  = cKDTree(xyz)
N     = len(xyz)

log(f"Points:  {N:,}")
gt_counts = np.bincount(labels, minlength=NUM_CLASSES)
log(f"GT label distribution:")
for c in range(NUM_CLASSES):
    if gt_counts[c] > 0:
        log(f"  {CLASS_NAMES[c]:<12s}: {gt_counts[c]:>10,}")

# ---------------------------------------------------------------------------
# CPU arrays + baseline RSS
# ---------------------------------------------------------------------------
prob_sum    = np.zeros((N, NUM_CLASSES), dtype=np.float32)
vote_count  = np.zeros(N, dtype=np.int32)
possibility = np.random.rand(N).astype(np.float64) * 1e-3

rss_before = current_rss_mb()
log(f"\nRSS before loop: {rss_before:.1f} MB")

t0     = time.time()
crop_n = 0

# ---------------------------------------------------------------------------
# Streaming inference loop
# ---------------------------------------------------------------------------
with torch.no_grad():
    while vote_count.min() < MIN_VOTES:
        pick_idx = int(np.argmin(possibility))
        center   = xyz[pick_idx]

        k_actual  = min(N_PTS, N)
        _, sel_idx = tree.query(center[np.newaxis], k=k_actual)
        sel_idx   = sel_idx[0].astype(np.int32)

        if k_actual < N_PTS:
            dup     = np.random.choice(k_actual, N_PTS - k_actual, replace=True)
            sel_idx = np.concatenate([sel_idx, sel_idx[dup]])

        sel_idx   = DP.shuffle_idx(sel_idx)
        sel_xyz   = xyz[sel_idx]
        sel_feats = feats[sel_idx]

        inputs  = build_inputs(sel_xyz, sel_feats, sel_idx)
        end_pts = net(inputs)
        _, end_pts = compute_loss(end_pts, cfg)

        probs = torch.softmax(end_pts['valid_logits'], dim=1).cpu().numpy()  # (N_PTS, 12)

        prob_sum[sel_idx]   += probs
        vote_count[sel_idx] += 1

        dists = np.sum((sel_xyz - center[np.newaxis]) ** 2, axis=1)
        delta = (1.0 - dists / (dists.max() + 1e-6)) ** 2
        possibility[sel_idx] += delta

        del inputs, end_pts, probs
        torch.cuda.empty_cache()

        crop_n += 1
        if crop_n % 10 == 0:
            uncovered = int((vote_count == 0).sum())
            log(f"  crop {crop_n:4d}  uncovered={uncovered:>8,}  "
                f"min_votes={vote_count.min()}  {time.time()-t0:.0f}s")

log(f"\nLoop complete: {crop_n} crops, {time.time()-t0:.0f}s")
rss_after_loop = current_rss_mb()
log(f"RSS after loop: {rss_after_loop:.1f} MB")

# Averaged probabilities and predictions
norm    = np.maximum(vote_count[:, np.newaxis].astype(np.float32), 1.0)
avg_prob = prob_sum / norm     # (N, 12)
pred     = avg_prob.argmax(axis=1)  # (N,)


# ===========================================================================
# CHECK 1: Coverage
# ===========================================================================
covered = int((vote_count > 0).sum())
cov_pct = 100.0 * covered / N
min_v   = int(vote_count.min())
mean_v  = float(vote_count.mean())
max_v   = int(vote_count.max())

log(f"\n{'─'*60}")
log(f"CHECK 1  Coverage")
log(f"  Points:  {N:,}")
log(f"  Covered: {covered:,}  ({cov_pct:.4f}%)")
log(f"  Votes:   min={min_v}  mean={mean_v:.1f}  max={max_v}")
log(f"  Crops:   {crop_n}")

passed = vote_count.min() >= MIN_VOTES and covered == N
log(f"  {'─'*40}")
log(f"  {PASS if passed else FAIL}: coverage {'100%' if passed else f'{cov_pct:.4f}% INCOMPLETE'}")
assert passed, f"Coverage failed: {N - covered} points uncovered"


# ===========================================================================
# CHECK 2: Prediction count
# ===========================================================================
log(f"\nCHECK 2  Prediction count")
log(f"  pred.shape: {pred.shape}")
log(f"  N:          {N}")

ok = (pred.shape == (N,))
log(f"  {PASS if ok else FAIL}: {'prediction count == N' if ok else f'shape mismatch {pred.shape} vs ({N},)'}")
assert ok


# ===========================================================================
# CHECK 3: Confusion matrix label integrity
# ===========================================================================
log(f"\nCHECK 3  Confusion matrix")

gt_classes   = sorted(np.unique(labels).tolist())
pred_classes = sorted(np.unique(pred).tolist())
log(f"  GT classes present:   {gt_classes}")
log(f"  Pred classes present: {pred_classes}")

# All GT labels in valid range
bad_gt = [c for c in gt_classes if c < 0 or c >= NUM_CLASSES]
assert not bad_gt, f"GT labels out of range: {bad_gt}"

# Build confusion matrix
cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
for c in range(NUM_CLASSES):
    for p in range(NUM_CLASSES):
        cm[c, p] = int(((labels == c) & (pred == p)).sum())

log(f"  CM shape: {cm.shape}")
log(f"  CM sum:   {cm.sum():,}  (expected {N:,})")

ok_shape = (cm.shape == (NUM_CLASSES, NUM_CLASSES))
ok_sum   = (cm.sum() == N)
assert ok_shape, f"CM shape {cm.shape} != ({NUM_CLASSES},{NUM_CLASSES})"
assert ok_sum,   f"CM sum {cm.sum()} != N={N}"

# Each GT class row must sum to its ground-truth count
row_ok = True
for c in gt_classes:
    row_sum  = int(cm[c].sum())
    gt_count = int(gt_counts[c])
    if row_sum != gt_count:
        log(f"  {FAIL}: class {c} ({CLASS_NAMES[c]}) row sum {row_sum} != gt {gt_count}")
        row_ok = False
assert row_ok

log(f"  {PASS}: CM shape {cm.shape}, sum={cm.sum():,}, all GT rows correct")


# ===========================================================================
# CHECK 4: No NaNs / Infs; row sums ≈ 1
# ===========================================================================
log(f"\nCHECK 4  NaN / Inf / row-sum check")

n_nan_ps  = int(np.isnan(prob_sum).sum())
n_nan_avg = int(np.isnan(avg_prob).sum())
n_inf_ps  = int(np.isinf(prob_sum).sum())
n_inf_avg = int(np.isinf(avg_prob).sum())

log(f"  prob_sum  NaN={n_nan_ps}  Inf={n_inf_ps}")
log(f"  avg_prob  NaN={n_nan_avg}  Inf={n_inf_avg}")

assert n_nan_ps == 0,  f"NaN in prob_sum"
assert n_nan_avg == 0, f"NaN in avg_prob"
assert n_inf_ps == 0,  f"Inf in prob_sum"
assert n_inf_avg == 0, f"Inf in avg_prob"

row_sums = avg_prob.sum(axis=1)
rs_min, rs_max = float(row_sums.min()), float(row_sums.max())
log(f"  avg_prob row sums: min={rs_min:.6f}  max={rs_max:.6f}  (expected ≈ 1.0)")

ok_rows = np.allclose(row_sums, 1.0, atol=1e-4)
log(f"  {PASS if ok_rows else FAIL}: row sums ≈ 1.0")
assert ok_rows, f"avg_prob row sums not ≈ 1.0: min={rs_min:.6f} max={rs_max:.6f}"


# ===========================================================================
# CHECK 5: Memory stable after freeing per-building arrays
# ===========================================================================
log(f"\nCHECK 5  Memory after freeing building arrays")
log(f"  RSS before loop: {rss_before:.1f} MB")
log(f"  RSS after loop:  {rss_after_loop:.1f} MB")

# Arrays freed: prob_sum (~91MB), vote_count (~7.6MB), possibility (~15MB),
#               xyz (~22MB), feats (~52MB), avg_prob (~91MB) → ~280 MB total
del prob_sum, vote_count, possibility, avg_prob, pred, norm
del xyz, feats, labels, tree, gt_counts
gc.collect()
torch.cuda.empty_cache()

rss_freed = current_rss_mb()
delta_loop   = rss_after_loop - rss_before
delta_freed  = rss_freed       - rss_before

log(f"  RSS after del+gc: {rss_freed:.1f} MB")
log(f"  Growth during loop: {delta_loop:+.1f} MB")
log(f"  Net growth vs start: {delta_freed:+.1f} MB  (tolerance: +500 MB)")

# Allow up to 500 MB net growth vs baseline (fragmentation + torch cache)
ok_mem = (delta_freed < 500.0)
log(f"  {PASS if ok_mem else FAIL}: net growth {delta_freed:+.1f} MB {'< 500 MB' if ok_mem else '>= 500 MB LEAK'}")
assert ok_mem, f"Memory leak: net growth {delta_freed:.0f} MB >= 500 MB"


# ===========================================================================
log(f"\n{'='*60}")
log(f"ALL 5 CHECKS PASSED")
log(f"Pipeline is correctly implemented.")
log(f"{'='*60}")
