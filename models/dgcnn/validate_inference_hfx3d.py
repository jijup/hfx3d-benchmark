#!/usr/bin/env python3
"""
Pipeline validation for DGCNN full-scene inference on HFX3D.

Uses randomly-initialised weights — checks pipeline mechanics, not accuracy.
Runs on HFX_BLD015 (smallest test building, ~1.9M pts).

Checks
------
  1. Coverage: ev.is_done() == True (vote_count.min() >= 1)
  2. Prediction count == N
  3. Confusion matrix shape/sum/GT-row integrity
  4. No NaNs/Infs in prob_sum or avg_prob; avg_prob row sums ≈ 1.0
  5. VmRSS net growth < 500 MB after freeing per-building arrays
"""
import os, sys, gc, time, types
import importlib.util
import numpy as np
import h5py
import torch

sys.path.insert(0, os.path.expanduser('~/DGCNN'))
from full_scene_infer import FullSceneEvaluator, compute_metrics

NUM_CLASSES = 12
CLASS_NAMES = [
    "wall", "window", "door", "balcony", "vegetation", "stairs",
    "terrain", "roof", "blinds", "other", "column", "arch",
]
BUILDING  = '/project/6118401/abdiaziz/data/HFX3D_Splits/test/HFX_BLD015_ZEB_CLEAN.h5'
CROP_SIZE = 4096
BATCH_SIZE = 16

PASS = '✓ PASS'
FAIL = '✗ FAIL'


def log(msg):
    print(msg, flush=True)


def current_rss_mb():
    try:
        with open('/proc/self/status') as fh:
            for line in fh:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return float('nan')


# ── Model (random weights) ────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location(
    'main_semseg_hfx3d',
    os.path.expanduser('~/DGCNN/main_semseg_hfx3d.py'),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
DGCNN_semseg_hfx3d = mod.DGCNN_semseg_hfx3d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Device: {device}")
model_args = types.SimpleNamespace(k=20, emb_dims=1024, dropout=0.5)
model = DGCNN_semseg_hfx3d(model_args).to(device)
model.eval()
log("Model: DGCNN with randomly-initialised weights (pipeline validation, not accuracy)")

# ── Load building ─────────────────────────────────────────────────────────
log(f"\nBuilding: {os.path.basename(BUILDING)}")
with h5py.File(BUILDING, 'r') as f:
    xyz    = f['point_cloud/xyz'][:].astype(np.float32)
    rgb    = f['point_cloud/rgb'][:]
    inten  = f['point_cloud/intensity'][:]
    labels = f['labels/semantic'][:].astype(np.int64)

feats = np.concatenate([xyz, rgb, inten[:, np.newaxis]], axis=1).astype(np.float32)
N = len(xyz)
log(f"Points: {N:,}")
gt_counts = np.bincount(labels, minlength=NUM_CLASSES)
log("GT label distribution:")
for c in range(NUM_CLASSES):
    if gt_counts[c] > 0:
        log(f"  {CLASS_NAMES[c]:<12s}: {gt_counts[c]:>10,}")

# ── Evaluator + baseline RSS ──────────────────────────────────────────────
ev = FullSceneEvaluator(xyz, num_classes=NUM_CLASSES, crop_size=CROP_SIZE, seed=0)
rss_before = current_rss_mb()
log(f"\nRSS before loop: {rss_before:.1f} MB")

t0 = time.time()
batch_count = 0

# ── Inference loop ────────────────────────────────────────────────────────
with torch.no_grad():
    while not ev.is_done():
        pending_sel   = []
        pending_crops = []
        for _ in range(BATCH_SIZE):
            if ev.is_done():
                break
            sel_idx, k_real, center_idx = ev.next_crop()
            crop = feats[sel_idx]   # (crop_size, 7) — no xyz centering for DGCNN
            pending_sel.append((sel_idx, k_real, center_idx))
            pending_crops.append(crop)

        batch = (
            torch.from_numpy(np.stack(pending_crops))
            .float()
            .permute(0, 2, 1)            # (B, 7, crop_size)
            .to(device)
        )
        logits = model(batch)                                       # (B, 12, crop_size)
        probs  = torch.softmax(logits, dim=1).permute(0, 2, 1).cpu().numpy()  # (B, crop_size, 12)

        for i, (sidx, kr, ci) in enumerate(pending_sel):
            ev.accumulate(ci, sidx, probs[i], kr)

        del batch, logits, probs
        torch.cuda.empty_cache()

        batch_count += 1
        if batch_count % 50 == 0:
            cov = ev.coverage_report()
            log(f"  batch {batch_count:4d}  coverage={cov['coverage_pct']:.1f}%  "
                f"uncovered={cov['total_points']-cov['covered_points']:,}  "
                f"{time.time()-t0:.0f}s")

log(f"\nLoop complete: {batch_count} batches, {time.time()-t0:.0f}s")
rss_after_loop = current_rss_mb()
log(f"RSS after loop: {rss_after_loop:.1f} MB")

# Averaged probabilities and predictions
counts   = np.maximum(ev.vote_count[:, np.newaxis].astype(np.float32), 1.0)
avg_prob = ev.prob_sum / counts
pred     = avg_prob.argmax(axis=1)


# =============================================================================
# CHECK 1: Coverage
# =============================================================================
cov     = ev.coverage_report()
covered = cov['covered_points']
cov_pct = cov['coverage_pct']
min_v   = cov['min_votes']
mean_v  = cov['mean_votes']
max_v   = cov['max_votes']

log(f"\n{'─'*60}")
log(f"CHECK 1  Coverage")
log(f"  Points:  {N:,}")
log(f"  Covered: {covered:,}  ({cov_pct:.4f}%)")
log(f"  Votes:   min={min_v}  mean={mean_v:.1f}  max={max_v}")
log(f"  Batches: {batch_count}")

passed = ev.is_done() and covered == N
log(f"  {'─'*40}")
log(f"  {PASS if passed else FAIL}: coverage {'100%' if passed else f'{cov_pct:.4f}% INCOMPLETE'}")
assert passed, f"Coverage failed: {N - covered} points uncovered"


# =============================================================================
# CHECK 2: Prediction count
# =============================================================================
log(f"\nCHECK 2  Prediction count")
log(f"  pred.shape: {pred.shape}")
log(f"  N:          {N}")

ok = (pred.shape == (N,))
log(f"  {PASS if ok else FAIL}: {'prediction count == N' if ok else f'shape mismatch {pred.shape} vs ({N},)'}")
assert ok


# =============================================================================
# CHECK 3: Confusion matrix
# =============================================================================
log(f"\nCHECK 3  Confusion matrix")

gt_classes   = sorted(np.unique(labels).tolist())
pred_classes = sorted(np.unique(pred).tolist())
log(f"  GT classes present:   {gt_classes}")
log(f"  Pred classes present: {pred_classes}")

bad_gt = [c for c in gt_classes if c < 0 or c >= NUM_CLASSES]
assert not bad_gt, f"GT labels out of range: {bad_gt}"

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

row_ok = True
for c in gt_classes:
    row_sum  = int(cm[c].sum())
    gt_count = int(gt_counts[c])
    if row_sum != gt_count:
        log(f"  {FAIL}: class {c} ({CLASS_NAMES[c]}) row sum {row_sum} != gt {gt_count}")
        row_ok = False
assert row_ok

log(f"  {PASS}: CM shape {cm.shape}, sum={cm.sum():,}, all GT rows correct")


# =============================================================================
# CHECK 4: NaN / Inf / row-sum
# =============================================================================
log(f"\nCHECK 4  NaN / Inf / row-sum check")

n_nan_ps  = int(np.isnan(ev.prob_sum).sum())
n_nan_avg = int(np.isnan(avg_prob).sum())
n_inf_ps  = int(np.isinf(ev.prob_sum).sum())
n_inf_avg = int(np.isinf(avg_prob).sum())

log(f"  prob_sum  NaN={n_nan_ps}  Inf={n_inf_ps}")
log(f"  avg_prob  NaN={n_nan_avg}  Inf={n_inf_avg}")

assert n_nan_ps == 0,  "NaN in prob_sum"
assert n_nan_avg == 0, "NaN in avg_prob"
assert n_inf_ps == 0,  "Inf in prob_sum"
assert n_inf_avg == 0, "Inf in avg_prob"

row_sums = avg_prob.sum(axis=1)
rs_min, rs_max = float(row_sums.min()), float(row_sums.max())
log(f"  avg_prob row sums: min={rs_min:.6f}  max={rs_max:.6f}  (expected ≈ 1.0)")

ok_rows = np.allclose(row_sums, 1.0, atol=1e-4)
log(f"  {PASS if ok_rows else FAIL}: row sums ≈ 1.0")
assert ok_rows, f"avg_prob row sums not ≈ 1.0: min={rs_min:.6f} max={rs_max:.6f}"


# =============================================================================
# CHECK 5: Memory
# =============================================================================
log(f"\nCHECK 5  Memory after freeing building arrays")
log(f"  RSS before loop: {rss_before:.1f} MB")
log(f"  RSS after loop:  {rss_after_loop:.1f} MB")

del avg_prob, pred, counts, cm, ev
del xyz, feats, labels, gt_counts
gc.collect()
torch.cuda.empty_cache()

rss_freed   = current_rss_mb()
delta_loop  = rss_after_loop - rss_before
delta_freed = rss_freed - rss_before

log(f"  RSS after del+gc: {rss_freed:.1f} MB")
log(f"  Growth during loop: {delta_loop:+.1f} MB")
log(f"  Net growth vs start: {delta_freed:+.1f} MB  (tolerance: +500 MB)")

ok_mem = (delta_freed < 500.0)
log(f"  {PASS if ok_mem else FAIL}: net growth {delta_freed:+.1f} MB {'< 500 MB' if ok_mem else '>= 500 MB LEAK'}")
assert ok_mem, f"Memory leak: net growth {delta_freed:.0f} MB >= 500 MB"


# =============================================================================
log(f"\n{'='*60}")
log(f"ALL 5 CHECKS PASSED — DGCNN pipeline is correctly implemented.")
log(f"{'='*60}")
