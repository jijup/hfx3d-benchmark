"""
HFX3D helper: config + data-processing utilities for RandLA-Net.

Avoids the C++ nearest_neighbors and cpp_subsampling extensions entirely.
Uses scipy.spatial.cKDTree for KNN and a pure-numpy grid subsampling.
"""

import numpy as np
import os, sys
from scipy.spatial import cKDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class ConfigHFX3D:
    k_n = 16                    # KNN neighbours
    num_layers = 5              # encoder/decoder depth (same as S3DIS)
    num_points = 45056          # points per crop (divisible by 4^4 * 2 = 512)
    num_classes = 12            # wall window door balcony veg stairs terrain roof blinds other column arch
    sub_grid_size = 0.04        # grid voxel for on-the-fly subsampling (m)

    batch_size = 4              # per-GPU during training
    val_batch_size = 4
    train_steps = 500
    val_steps = 100

    sub_sampling_ratio = [4, 4, 4, 4, 2]   # encoder downsampling at each layer
    d_out = [16, 64, 128, 256, 512]         # feature dims per layer

    noise_init = 3.5
    max_epoch = 200
    learning_rate = 1e-2
    lr_decays = {i: 0.95 for i in range(0, 500)}

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None

    # Set externally by dataset __init__
    ignored_label_inds = []
    class_weights = None

    # Input feature dimension: xyz(3) + rgb(3) + intensity(1) = 7
    input_feature_dim = 7


# ---------------------------------------------------------------------------
# Data Processing
# ---------------------------------------------------------------------------

class DataProcessing:

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """
        Batched KNN using scipy cKDTree.

        Parameters
        ----------
        support_pts : (B, N1, 3)
        query_pts   : (B, N2, 3)
        k           : int

        Returns
        -------
        neighbor_idx : (B, N2, k)  int32
        """
        B, N2, _ = query_pts.shape
        neighbor_idx = np.zeros((B, N2, k), dtype=np.int32)
        for b in range(B):
            tree = cKDTree(support_pts[b])
            _, idx = tree.query(query_pts[b], k=k, workers=-1)
            idx = np.atleast_2d(idx) if k == 1 else idx
            if k == 1:
                idx = idx.reshape(N2, 1)
            neighbor_idx[b] = idx.astype(np.int32)
        return neighbor_idx

    @staticmethod
    def grid_sub_sampling(points, features=None, labels=None, grid_size=0.04):
        """
        Voxel-grid subsampling (barycenter averaging).

        Parameters
        ----------
        points   : (N, 3) float32
        features : (N, d) float32  or None
        labels   : (N,) int        or None
        grid_size: float

        Returns
        -------
        sub_points[, sub_features][, sub_labels]
        """
        mins = np.floor(points.min(axis=0) / grid_size).astype(np.int64)
        voxel_idx = np.floor(points / grid_size).astype(np.int64) - mins
        keys = (voxel_idx[:, 0] * int(1e8)
                + voxel_idx[:, 1] * int(1e4)
                + voxel_idx[:, 2])
        order = np.argsort(keys)
        keys_sorted = keys[order]
        _, first_occ, counts = np.unique(keys_sorted,
                                         return_index=True,
                                         return_counts=True)

        sub_points = np.zeros((len(first_occ), 3), dtype=np.float32)
        for i, (start, cnt) in enumerate(zip(first_occ, counts)):
            sub_points[i] = points[order[start:start + cnt]].mean(axis=0)

        results = [sub_points]

        if features is not None:
            sub_features = np.zeros((len(first_occ), features.shape[1]),
                                    dtype=np.float32)
            for i, (start, cnt) in enumerate(zip(first_occ, counts)):
                sub_features[i] = features[order[start:start + cnt]].mean(axis=0)
            results.append(sub_features)

        if labels is not None:
            sub_labels = np.zeros(len(first_occ), dtype=labels.dtype)
            for i, (start, cnt) in enumerate(zip(first_occ, counts)):
                chunk = labels[order[start:start + cnt]]
                sub_labels[i] = np.bincount(chunk.astype(np.int64)).argmax()
            results.append(sub_labels)

        return tuple(results) if len(results) > 1 else results[0]

    @staticmethod
    def shuffle_idx(x):
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(len(data_list))
        np.random.shuffle(indices)
        return [data_list[i] for i in indices]

    @staticmethod
    def get_class_weights(num_per_class):
        """
        Compute inverse-frequency weights from per-class point counts.

        Parameters
        ----------
        num_per_class : array-like of length num_classes

        Returns
        -------
        (1, num_classes) float32
        """
        num_per_class = np.array(num_per_class, dtype=np.float64)
        weight = num_per_class / float(num_per_class.sum())
        ce_label_weight = 1.0 / (weight + 0.02)
        return np.expand_dims(ce_label_weight.astype(np.float32), axis=0)

    @staticmethod
    def IoU_from_confusions(confusions):
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)
        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)
        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)
        IoU += mask * mIoU
        return IoU
