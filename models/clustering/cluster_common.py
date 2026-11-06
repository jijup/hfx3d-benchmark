"""
Shared helpers for the HFX3D oracle-semantic clustering baselines.

Clustering runs per thing class on GROUND-TRUTH semantics, on 5 cm voxels of
that class's points. A voxel's feature is the mean 7-D feature
[x, y, z, r, g, b, intensity] of its points, standardised with per-class
statistics computed once over the TRAINING split, so a given radius means the
same physical distance in every building.
"""

import glob
import os

import h5py
import numpy as np

CLASS_NAMES = [
    "wall", "window", "door", "balcony", "vegetation", "stairs",
    "terrain", "roof", "blinds", "other", "column", "arch",
]
THING_CLASSES = [1, 2, 3, 4, 5, 8, 10, 11]
SEGMENT_IGNORE_INDEX = (-1, 0, 6, 7, 9)  # ignore + stuff (wall, terrain, roof, other)
INSTANCE_IGNORE_INDEX = -1
VOXEL_SIZE = 0.05  # metres, same as the learned instance methods
SEED = 42


def list_buildings(data_root, split):
    return sorted(glob.glob(os.path.join(data_root, split, "*.h5")))


def load_building(path, with_instance=True):
    """Returns xyz, feat (N, 7) = [xyz, rgb, intensity], semantic, instance."""
    with h5py.File(path, "r") as f:
        xyz = f["point_cloud/xyz"][:].astype(np.float32)
        rgb = f["point_cloud/rgb"][:].astype(np.float32)
        intensity = f["point_cloud/intensity"][:].astype(np.float32)
        semantic = f["labels/semantic"][:].astype(np.int64)
        instance = f["labels/instance"][:].astype(np.int64) if with_instance else None
    feat = np.concatenate([xyz, rgb, intensity[:, None]], axis=1)
    return xyz, feat, semantic, instance


def voxelize(xyz, feat, voxel_size=VOXEL_SIZE):
    """Group points into voxels.

    Returns (voxel_feat (V, 7) = mean feature per voxel, point_to_voxel (N,)).
    Deterministic: voxels are ordered by their integer grid key.
    """
    grid = np.floor(xyz / voxel_size).astype(np.int64)
    grid -= grid.min(axis=0)
    dims = grid.max(axis=0) + 1
    key = (grid[:, 0] * dims[1] + grid[:, 1]) * dims[2] + grid[:, 2]
    _, point_to_voxel, counts = np.unique(key, return_inverse=True, return_counts=True)
    voxel_feat = np.stack(
        [np.bincount(point_to_voxel, weights=feat[:, c], minlength=len(counts)) for c in range(feat.shape[1])],
        axis=1,
    ) / counts[:, None]
    return voxel_feat.astype(np.float32), point_to_voxel


def instance_sizes_in_voxels(instance, point_to_voxel):
    """Number of distinct voxels covered by each GT instance (ignore -1 excluded)."""
    valid = instance != INSTANCE_IGNORE_INDEX
    if not valid.any():
        return np.zeros(0, dtype=np.int64)
    n_vox = int(point_to_voxel.max()) + 1
    pair = np.unique(instance[valid] * n_vox + point_to_voxel[valid])
    _, sizes = np.unique(pair // n_vox, return_counts=True)
    return sizes


class LazyMasks:
    """Dense per-instance masks built on demand from point-index arrays, so a
    building with thousands of predicted instances never holds them all in
    memory at once (the evaluator only reads one mask at a time)."""

    def __init__(self, index_arrays, n_points):
        self.index_arrays = index_arrays
        self.n_points = n_points

    def __len__(self):
        return len(self.index_arrays)

    def __getitem__(self, i):
        mask = np.zeros(self.n_points, dtype=bool)
        mask[self.index_arrays[i]] = True
        return mask
