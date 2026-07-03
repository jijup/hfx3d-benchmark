#!/usr/bin/env python3
"""
RandLA-Net full-scene benchmark inference for HFX3D.

Three clearly separated inference regimes
-----------------------------------------
  Training      → main_HFX3D.py  (stochastic multi-crop, TRAIN_CROPS_PER_CLOUD=50)
  Validation    → main_HFX3D.py  (deterministic fixed-budget, VAL_CROPS_PER_CLOUD=20,
                                   reset_possibilities each epoch, num_workers=0)
  Final testing → THIS SCRIPT    (streaming, adaptive, stops per-building when
                                   min(vote_count) >= --min_votes)

Final-test streaming loop (per building)
-----------------------------------------
  1.  Load xyz / features / labels once from HDF5
  2.  Build cKDTree once
  3.  Allocate vote_count (int32), prob_sum (float32), possibility (float64) on CPU
  4.  Pick crop centre = point with lowest possibility
  5.  KNN query → sel_idx (N_PTS unique point indices)
  6.  Build multi-scale inputs, move crop tensors to GPU
  7.  Forward pass
  8.  Softmax probs → CPU numpy
  9.  Accumulate: prob_sum[sel_idx] += probs; vote_count[sel_idx] += 1
  10. Update possibility; discard GPU tensors
  11. Repeat 4-10 until vote_count.min() >= min_votes
  12. Hard assertion before metrics
  13. Free per-building arrays; continue to next building

Memory layout
-------------
  CPU: prob_sum, vote_count, possibility, xyz, feats, labels (one building at a time)
  GPU: current crop only (build_inputs result + forward pass intermediates)
"""
import os
import glob
import argparse
import time
import numpy as np
import h5py
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

from helper_hfx3d import ConfigHFX3D as cfg, DataProcessing as DP
from hfx3d_dataset import NUM_CLASSES, LABEL_TO_NAMES
from RandLANet_HFX3D import Network, compute_loss

CLASS_NAMES = [LABEL_TO_NAMES[i] for i in range(NUM_CLASSES)]
N_PTS = cfg.num_points  # 45056 — points per crop

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data_root',
                    default='/project/6118401/abdiaziz/data/HFX3D_Splits')
parser.add_argument('--checkpoint',
                    required=True,
                    help='Path to checkpoint_best.tar')
parser.add_argument('--min_votes', type=int, default=1,
                    help='Stop per building once every point has >= this many votes')
parser.add_argument('--log_every', type=int, default=50,
                    help='Log progress every N crops')
parser.add_argument('--seed', type=int, default=0)
FLAGS = parser.parse_args()

np.random.seed(FLAGS.seed)


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Network(cfg)
net.to(device)

if not os.path.isfile(FLAGS.checkpoint):
    raise FileNotFoundError(f"Checkpoint not found: {FLAGS.checkpoint}")

ckpt = torch.load(FLAGS.checkpoint, map_location=device)
net.load_state_dict(ckpt['model_state_dict'])
log(f"Checkpoint:  {FLAGS.checkpoint}")
log(f"Epoch:       {ckpt.get('epoch', '?')}")
log(f"Best mIoU:   {ckpt.get('miou', float('nan'))*100:.2f}%")

if torch.cuda.device_count() > 1:
    log(f"Using {torch.cuda.device_count()} GPUs")
    net = nn.DataParallel(net)

net.eval()


# ---------------------------------------------------------------------------
# Build multi-scale inputs for one crop (mirrors hfx3d_dataset.tf_map)
# ---------------------------------------------------------------------------
def build_inputs(sel_xyz, sel_feats, sel_idx):
    """
    sel_xyz   : (N_PTS, 3)  float32   crop point coordinates
    sel_feats : (N_PTS, 7)  float32   per-point features [x,y,z,r,g,b,i]
    sel_idx   : (N_PTS,)    int32     indices into the building point cloud

    Returns a dict with all tensors already on GPU.
    Dummy labels (all 0) are used — they won't affect forward logits since
    cfg.ignored_label_inds is empty.
    """
    batch_xyz   = sel_xyz[np.newaxis]          # (1, N_PTS, 3)
    batch_feats = sel_feats[np.newaxis]        # (1, N_PTS, 7)
    batch_label = np.zeros((1, N_PTS), dtype=np.int32)
    batch_idx   = sel_idx[np.newaxis]          # (1, N_PTS)
    batch_cind  = np.zeros((1, 1), dtype=np.int32)

    input_points, input_neighbors, input_pools, input_up_samples = [], [], [], []
    current_xyz = batch_xyz.copy()

    for i in range(cfg.num_layers):
        neighbour_idx = DP.knn_search(current_xyz, current_xyz, cfg.k_n)
        n_sub = current_xyz.shape[1] // cfg.sub_sampling_ratio[i]
        sub_points = current_xyz[:, :n_sub, :]
        pool_i = neighbour_idx[:, :n_sub, :]
        up_i = DP.knn_search(sub_points, current_xyz, 1)

        input_points.append(current_xyz)
        input_neighbors.append(neighbour_idx)
        input_pools.append(pool_i)
        input_up_samples.append(up_i)

        current_xyz = sub_points

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
# Load one building from HDF5
# ---------------------------------------------------------------------------
def load_building(fp):
    with h5py.File(fp, 'r') as f:
        xyz    = f['point_cloud/xyz'][:]        # (N, 3) float32
        rgb    = f['point_cloud/rgb'][:]        # (N, 3) float32
        inten  = f['point_cloud/intensity'][:]  # (N,)   float32
        labels = f['labels/semantic'][:]        # (N,)   uint8
    feats = np.concatenate(
        [xyz, rgb, inten[:, np.newaxis]], axis=1
    ).astype(np.float32)                        # (N, 7)
    tree = cKDTree(xyz)
    return xyz.astype(np.float32), feats, labels.astype(np.int64), tree


# ---------------------------------------------------------------------------
# Discover test buildings
# ---------------------------------------------------------------------------
test_dir = os.path.join(FLAGS.data_root, 'test')
building_files = sorted(glob.glob(os.path.join(test_dir, '*.h5')))
assert building_files, f"No .h5 files found in {test_dir}"
log(f"\nTest split: {len(building_files)} buildings  ({test_dir})")
log(f"Min votes required: {FLAGS.min_votes}")

# Accumulators across all buildings
all_tp = np.zeros(NUM_CLASSES, dtype=np.int64)
all_fp = np.zeros(NUM_CLASSES, dtype=np.int64)
all_fn = np.zeros(NUM_CLASSES, dtype=np.int64)

t_total = time.time()

# ---------------------------------------------------------------------------
# Per-building streaming inference
# ---------------------------------------------------------------------------
for fp in building_files:
    bname = os.path.basename(fp)
    log(f"\n{'='*64}")
    log(f"Building: {bname}")

    xyz, feats, labels, tree = load_building(fp)
    N = len(xyz)
    log(f"  Points:  {N:,}")

    # CPU accumulators — only the current crop moves to GPU
    prob_sum   = np.zeros((N, NUM_CLASSES), dtype=np.float32)  # vote-averaged probs
    vote_count = np.zeros(N, dtype=np.int32)                   # votes per point
    possibility = np.random.rand(N).astype(np.float64) * 1e-3  # sampling priority

    t0 = time.time()
    crop_n = 0

    with torch.no_grad():
        while vote_count.min() < FLAGS.min_votes:
            # 1. Pick centre with lowest possibility
            pick_idx = int(np.argmin(possibility))
            center   = xyz[pick_idx]              # (3,)

            # 2. KNN crop (unique indices from tree query)
            k_actual = min(N_PTS, N)
            _, sel_idx = tree.query(center[np.newaxis], k=k_actual)
            sel_idx = sel_idx[0].astype(np.int32)  # (k_actual,) — guaranteed unique

            # Pad to N_PTS if building is smaller (edge case; unlikely for test set)
            if k_actual < N_PTS:
                dup = np.random.choice(k_actual, N_PTS - k_actual, replace=True)
                sel_idx = np.concatenate([sel_idx, sel_idx[dup]])

            sel_idx = DP.shuffle_idx(sel_idx)          # (N_PTS,)
            sel_xyz   = xyz[sel_idx]                   # (N_PTS, 3)
            sel_feats = feats[sel_idx]                 # (N_PTS, 7)

            # 3. Build multi-scale inputs → GPU
            inputs = build_inputs(sel_xyz, sel_feats, sel_idx)

            # 4. Forward pass
            end_pts = net(inputs)
            _, end_pts = compute_loss(end_pts, cfg)

            # 5. Softmax probabilities → CPU  (N_PTS, NUM_CLASSES)
            probs = (
                torch.softmax(end_pts['valid_logits'], dim=1)
                .cpu()
                .numpy()
            )  # (N_PTS, 12) — all points valid since ignored_label_inds=[]

            # 6. Accumulate into CPU arrays
            #    sel_idx is unique (KNN + shuffle) so simple += is correct
            prob_sum[sel_idx]   += probs
            vote_count[sel_idx] += 1

            # 7. Update possibility so under-covered regions are sampled next
            dists = np.sum((sel_xyz - center[np.newaxis]) ** 2, axis=1)
            delta = (1.0 - dists / (dists.max() + 1e-6)) ** 2
            possibility[sel_idx] += delta

            # 8. Discard GPU tensors
            del inputs, end_pts, probs
            torch.cuda.empty_cache()

            crop_n += 1

            if crop_n % FLAGS.log_every == 0:
                uncovered = int((vote_count == 0).sum())
                elapsed   = time.time() - t0
                log(f"  crop {crop_n:5d}  uncovered={uncovered:>10,}  "
                    f"min_votes={vote_count.min()}  elapsed={elapsed:.0f}s")

    # Coverage statistics
    min_v   = int(vote_count.min())
    mean_v  = float(vote_count.mean())
    max_v   = int(vote_count.max())
    covered = int((vote_count > 0).sum())
    cov_pct = 100.0 * covered / N
    elapsed = time.time() - t0

    log(f"\n  Coverage:  {cov_pct:.2f}%  ({covered:,} / {N:,} points)")
    log(f"  Votes:     min={min_v}  mean={mean_v:.1f}  max={max_v}")
    log(f"  Crops:     {crop_n}")
    log(f"  Time:      {elapsed:.0f}s")

    # Hard assertion — mandatory before metric computation
    assert vote_count.min() >= FLAGS.min_votes, (
        f"COVERAGE FAILED for {bname}: "
        f"min_votes={vote_count.min()} < {FLAGS.min_votes}  "
        f"({covered:,}/{N:,} points, {cov_pct:.2f}%)"
    )
    log(f"  Coverage assertion passed.")

    # Predict: argmax of averaged softmax probabilities
    norm = np.maximum(vote_count[:, np.newaxis].astype(np.float32), 1.0)
    pred = (prob_sum / norm).argmax(axis=1)   # (N,)

    # Per-class TP / FP / FN
    for c in range(NUM_CLASSES):
        all_tp[c] += int(((pred == c) & (labels == c)).sum())
        all_fp[c] += int(((pred == c) & (labels != c)).sum())
        all_fn[c] += int(((pred != c) & (labels == c)).sum())

    # Running mIoU after this building
    denom_b = (all_tp + all_fp + all_fn).astype(np.float64)
    miou_b  = float(np.where(denom_b > 0, all_tp / (denom_b + 1e-10), 0.0).mean())
    log(f"  Running mIoU (cumulative): {miou_b*100:.2f}%")

    # Free per-building arrays
    del prob_sum, vote_count, possibility, xyz, feats, labels, tree

# ---------------------------------------------------------------------------
# Final benchmark metrics
# ---------------------------------------------------------------------------
denom_iou = (all_tp + all_fp + all_fn).astype(np.float64)
iou  = np.where(denom_iou > 0, all_tp / (denom_iou + 1e-10), 0.0)
prec = np.where(all_tp + all_fp > 0, all_tp / (all_tp + all_fp + 1e-10), 0.0)
rec  = np.where(all_tp + all_fn > 0, all_tp / (all_tp + all_fn + 1e-10), 0.0)
f1   = np.where(prec + rec > 0, 2.0 * prec * rec / (prec + rec + 1e-10), 0.0)
oa   = float(all_tp.sum()) / float((all_tp + all_fn).sum() + 1e-9)

log("\n" + "="*64)
log("FINAL TEST RESULTS (all buildings combined)")
log("="*64)
log(f"OA:   {oa*100:.2f}%")
log(f"mIoU: {iou.mean()*100:.2f}%")
log(f"mP:   {prec.mean()*100:.2f}%")
log(f"mR:   {rec.mean()*100:.2f}%")
log(f"mF1:  {f1.mean()*100:.2f}%")
log(f"\nTotal inference time: {(time.time()-t_total)/60:.1f} min")
log("\nPer-class results:")
log(f"{'Class':<12s}  {'IoU':>7s}  {'P':>7s}  {'R':>7s}  {'F1':>7s}")
log("-" * 46)
for c in range(NUM_CLASSES):
    log(f"{CLASS_NAMES[c]:<12s}  "
        f"{iou[c]*100:6.2f}%  "
        f"{prec[c]*100:6.2f}%  "
        f"{rec[c]*100:6.2f}%  "
        f"{f1[c]*100:6.2f}%")
