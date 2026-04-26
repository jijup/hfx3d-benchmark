"""
HFX3D dataset adapter for RandLA-Net (PyTorch).

Each building is an HDF5 file with:
  point_cloud/xyz        (N, 3) float32  -- unit-sphere normalised
  point_cloud/rgb        (N, 3) float32  -- [0, 1]
  point_cloud/intensity  (N,)   float32  -- [0, 1]
  labels/semantic        (N,)   uint8    -- 0-11

12 classes
  0=wall  1=window  2=door    3=balcony   4=vegetation  5=stairs
  6=terrain  7=roof  8=blinds  9=other  10=column  11=arch

Input feature vector fed to the network: [x, y, z, r, g, b, intensity]
(7-D).  The network first-layer conv maps this to 8 channels.
"""

import os
import glob
import numpy as np
import h5py
import torch
import torch.utils.data as torch_data

from helper_hfx3d import ConfigHFX3D as cfg, DataProcessing as DP
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Label definitions
# ---------------------------------------------------------------------------

LABEL_TO_NAMES = {
    0:  'wall',
    1:  'window',
    2:  'door',
    3:  'balcony',
    4:  'vegetation',
    5:  'stairs',
    6:  'terrain',
    7:  'roof',
    8:  'blinds',
    9:  'other',
    10: 'column',
    11: 'arch',
}

NUM_CLASSES = len(LABEL_TO_NAMES)


# Approximate per-class point counts across HFX3D (placeholder; will be
# recomputed from training data at dataset __init__ if desired, but these
# reasonable priors avoid division-by-zero and produce sensible weights).
PRIOR_COUNTS = np.array(
    [500000, 80000, 60000, 40000, 120000, 20000,
     200000, 300000, 10000, 50000, 15000, 5000],
    dtype=np.float64,
)

# Crops per building per epoch.  Mirrors the original train_steps/val_steps
# behaviour: each "epoch" is a fixed number of random crops, not one pass
# through the buildings.  13*50=650 train items, 5*20=100 val items.
TRAIN_CROPS_PER_CLOUD = 50
VAL_CROPS_PER_CLOUD   = 20


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HFX3DDataset(torch_data.Dataset):
    """
    RandLA-Net compatible dataset for HFX3D.

    Parameters
    ----------
    mode : 'training' | 'validation'
    data_root : str   path to WACV_Splits/
    num_points : int  points per crop (default cfg.num_points)
    """

    def __init__(self, mode,
                 data_root='/home/abdiaziz/projects/def-jijup/abdiaziz/WACV_Splits',
                 num_points=None):
        self.name = 'HFX3D'
        self.mode = mode
        self.num_points = num_points or cfg.num_points

        split = 'train' if mode == 'training' else 'test'
        split_dir = os.path.join(data_root, split)
        self.data_list = sorted(glob.glob(os.path.join(split_dir, '*.h5')))
        assert len(self.data_list) > 0, \
            f"No .h5 files found in {split_dir}"

        # Build KD-trees for every cloud (loaded lazily)
        self._cache = {}   # file_path -> (points, features, labels, cKDTree)

        # Class weights from prior counts
        cfg.ignored_label_inds = []
        cfg.class_weights = DP.get_class_weights(PRIOR_COUNTS)

        # For test-time possibility sampling
        if mode != 'training':
            self.possibility = []
            self.min_possibility = []
            for fp in self.data_list:
                pts, _, _, _ = self._load_cloud(fp)
                N = pts.shape[0]
                p = np.random.rand(N) * 1e-3
                self.possibility.append(p)
                self.min_possibility.append(float(np.min(p)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_cloud(self, file_path):
        """Load + cache a building cloud.  Returns (xyz, feats, labels, tree)."""
        if file_path in self._cache:
            return self._cache[file_path]

        with h5py.File(file_path, 'r') as f:
            xyz   = f['point_cloud/xyz'][:]        # (N,3) float32
            rgb   = f['point_cloud/rgb'][:]        # (N,3) float32
            inten = f['point_cloud/intensity'][:]  # (N,)  float32
            if self.mode != 'training' and 'labels/semantic' not in f:
                labels = np.zeros(xyz.shape[0], dtype=np.uint8)
            else:
                labels = f['labels/semantic'][:]   # (N,)  uint8

        feats = np.concatenate(
            [xyz,
             rgb,
             inten[:, np.newaxis]],
            axis=1,
        ).astype(np.float32)   # (N, 7)

        tree = cKDTree(xyz)
        self._cache[file_path] = (xyz, feats, labels, tree)
        return xyz, feats, labels, tree

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        if self.mode == 'training':
            return len(self.data_list) * TRAIN_CROPS_PER_CLOUD
        else:
            return len(self.data_list) * VAL_CROPS_PER_CLOUD

    def __getitem__(self, item):
        pc, feats, labels, idx, cloud_ind = self._spatially_regular_gen(item)
        return pc, feats, labels, idx, cloud_ind

    def _spatially_regular_gen(self, item):
        if self.mode == 'training':
            cloud_ind = item % len(self.data_list)
            fp = self.data_list[cloud_ind]
            xyz, feats, labels, tree = self._load_cloud(fp)
            pick_idx = np.random.choice(len(xyz), 1)
        else:
            cloud_ind = int(np.argmin(self.min_possibility))
            fp = self.data_list[cloud_ind]
            xyz, feats, labels, tree = self._load_cloud(fp)
            pick_idx = np.array([np.argmin(self.possibility[cloud_ind])])

        # Crop a sphere / KNN neighbourhood of num_points
        center = xyz[pick_idx, :]             # (1, 3)
        k_actual = min(self.num_points, len(xyz))
        _, sel_idx = tree.query(center, k=k_actual)
        sel_idx = sel_idx[0]                  # (k_actual,)

        # Pad if cloud is smaller than num_points
        if k_actual < self.num_points:
            dup = np.random.choice(k_actual, self.num_points - k_actual)
            sel_idx = np.concatenate([sel_idx, sel_idx[dup]])

        sel_idx = DP.shuffle_idx(sel_idx)
        sel_xyz   = xyz[sel_idx].astype(np.float32)
        sel_feats = feats[sel_idx].astype(np.float32)
        sel_labels = labels[sel_idx].astype(np.int32)

        if self.mode != 'training':
            dists = np.sum(np.square(sel_xyz - xyz[pick_idx]).astype(np.float32),
                           axis=1)
            delta = np.square(1 - dists / (np.max(dists) + 1e-6))
            self.possibility[cloud_ind][sel_idx] += delta
            self.min_possibility[cloud_ind] = float(
                np.min(self.possibility[cloud_ind]))

        return (sel_xyz,
                sel_feats,
                sel_labels,
                sel_idx.astype(np.int32),
                np.array([cloud_ind], dtype=np.int32))

    # ------------------------------------------------------------------
    # Collate + tf_map
    # ------------------------------------------------------------------

    def tf_map(self, batch_xyz, batch_feats, batch_label,
               batch_pc_idx, batch_cloud_idx):
        """
        Build multi-layer inputs required by RandLA-Net.

        batch_xyz   : (B, N, 3)
        batch_feats : (B, N, 7)
        """
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        current_xyz = batch_xyz.copy()

        for i in range(cfg.num_layers):
            neighbour_idx = DP.knn_search(current_xyz, current_xyz, cfg.k_n)
            n_sub = current_xyz.shape[1] // cfg.sub_sampling_ratio[i]
            sub_points = current_xyz[:, :n_sub, :]
            pool_i  = neighbour_idx[:, :n_sub, :]
            up_i    = DP.knn_search(sub_points, current_xyz, 1)

            input_points.append(current_xyz)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)

            current_xyz = sub_points

        inputs = {}
        inputs['xyz']        = [torch.from_numpy(p).float() for p in input_points]
        inputs['neigh_idx']  = [torch.from_numpy(n).long()  for n in input_neighbors]
        inputs['sub_idx']    = [torch.from_numpy(p).long()  for p in input_pools]
        inputs['interp_idx'] = [torch.from_numpy(u).long()  for u in input_up_samples]
        # features: (B, C, N)
        inputs['features']   = torch.from_numpy(
            batch_feats).float().permute(0, 2, 1)   # (B, 7, N)
        inputs['labels']     = torch.from_numpy(batch_label).long()
        inputs['input_inds'] = torch.from_numpy(batch_pc_idx).long()
        inputs['cloud_inds'] = torch.from_numpy(batch_cloud_idx).long()
        return inputs

    def collate_fn(self, batch):
        sel_xyz, sel_feats, sel_labels, sel_idx, cloud_ind = zip(*batch)

        batch_xyz    = np.stack(sel_xyz)    # (B, N, 3)
        batch_feats  = np.stack(sel_feats)  # (B, N, 7)
        batch_labels = np.stack(sel_labels) # (B, N)
        batch_idx    = np.stack(sel_idx)    # (B, N)
        batch_cind   = np.stack(cloud_ind)  # (B, 1)

        return self.tf_map(batch_xyz, batch_feats, batch_labels,
                           batch_idx, batch_cind)
