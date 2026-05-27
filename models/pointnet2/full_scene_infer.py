"""
Full-scene inference utility for DGCNN and PointNet++ on HFX3D.

Provides possibility-based KNN crop sampling (RandLA-Net style) that
guarantees 100% point coverage before yielding predictions, and a
compute_metrics helper that returns all five benchmark metrics.
"""

import numpy as np
from scipy.spatial import cKDTree


class FullSceneEvaluator:
    """
    Iteratively samples KNN crops centred on the lowest-possibility point,
    accumulates per-point softmax probability sums, and stops when every
    point has received at least one vote.
    """

    def __init__(self, xyz, num_classes=12, crop_size=4096, seed=0):
        rng = np.random.default_rng(seed)
        N = len(xyz)
        self.xyz = xyz.astype(np.float32)
        self.crop_size = crop_size
        self.num_classes = num_classes
        # Small random init so no two points share the same priority.
        self.possibility = rng.random(N).astype(np.float32) * 1e-3
        self.prob_sum = np.zeros((N, num_classes), dtype=np.float32)
        self.vote_count = np.zeros(N, dtype=np.int32)
        self._tree = cKDTree(self.xyz)
        # Stores original possibility values for centres currently in-flight
        # (set to inf so the same centre is not re-chosen within a batch).
        self._blocked = {}

    def is_done(self):
        return bool(self.vote_count.min() > 0)

    def next_crop(self):
        """
        Returns (sel_idx, k_real, center_idx).
        sel_idx: int64 array of length crop_size (may contain padding repeats).
        k_real:  number of distinct points at the start of sel_idx.
        center_idx: index of the crop centre (blocked until accumulate is called).
        """
        center_idx = int(np.argmin(self.possibility))
        self._blocked[center_idx] = float(self.possibility[center_idx])
        self.possibility[center_idx] = np.inf

        N = len(self.xyz)
        k_real = min(self.crop_size, N)
        _, idx = self._tree.query(self.xyz[center_idx], k=k_real)
        idx = np.asarray(idx, dtype=np.int64)

        if k_real < self.crop_size:
            extra = np.random.choice(k_real, self.crop_size - k_real, replace=True)
            sel_idx = np.concatenate([idx, idx[extra]])
        else:
            sel_idx = idx

        return sel_idx, k_real, center_idx

    def accumulate(self, center_idx, sel_idx, probs, k_real):
        """
        probs: (crop_size, num_classes) — softmax probabilities (NOT log-softmax).
        Only the first k_real entries correspond to distinct real points;
        padding entries (if any) are ignored.
        """
        real_idx = sel_idx[:k_real]
        self.prob_sum[real_idx] += probs[:k_real]
        self.vote_count[real_idx] += 1

        # Restore centre possibility then boost all covered points so they are
        # deprioritised in future crops (RandLA-Net quadratic weighting).
        self.possibility[center_idx] = self._blocked.pop(center_idx, 0.0)
        center_xyz = self.xyz[center_idx]
        dists_sq = np.sum((self.xyz[real_idx] - center_xyz) ** 2, axis=1)
        max_d2 = max(float(dists_sq.max()), 1e-8)
        self.possibility[real_idx] += (1.0 - dists_sq / max_d2).astype(np.float32)

    def predictions(self):
        if not self.is_done():
            raise RuntimeError("Not all points covered — run inference until is_done()")
        counts = np.maximum(self.vote_count[:, np.newaxis].astype(np.float32), 1.0)
        return (self.prob_sum / counts).argmax(axis=1).astype(np.int32)

    def coverage_report(self):
        total = len(self.vote_count)
        covered = int((self.vote_count > 0).sum())
        return {
            "total_points": total,
            "covered_points": covered,
            "coverage_pct": 100.0 * covered / total,
            "min_votes": int(self.vote_count.min()),
            "mean_votes": float(self.vote_count.mean()),
            "max_votes": int(self.vote_count.max()),
        }


def compute_metrics(pred, gt, num_classes):
    """
    Compute mIoU, OA, mP, mR, mF1 from full-scene per-point predictions.
    Points with gt == -1 are ignored.
    Returns a dict with scalar metrics and per-class arrays.
    """
    mask = gt >= 0
    pred = pred[mask]
    gt = gt[mask]

    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    for c in range(num_classes):
        tp[c] = int(((pred == c) & (gt == c)).sum())
        fp[c] = int(((pred == c) & (gt != c)).sum())
        fn[c] = int(((pred != c) & (gt == c)).sum())

    denom_iou = (tp + fp + fn).astype(np.float64)
    iou = np.where(denom_iou > 0, tp / (denom_iou + 1e-10), 0.0)

    denom_p = (tp + fp).astype(np.float64)
    precision = np.where(denom_p > 0, tp / (denom_p + 1e-10), 0.0)

    denom_r = (tp + fn).astype(np.float64)
    recall = np.where(denom_r > 0, tp / (denom_r + 1e-10), 0.0)

    f1 = np.where(
        precision + recall > 0,
        2.0 * precision * recall / (precision + recall + 1e-10),
        0.0,
    )

    oa = float(tp.sum()) / float(len(gt)) if len(gt) > 0 else 0.0

    return {
        "mIoU": float(iou.mean()),
        "OA": oa,
        "mP": float(precision.mean()),
        "mR": float(recall.mean()),
        "mF1": float(f1.mean()),
        "per_class_iou": iou,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
