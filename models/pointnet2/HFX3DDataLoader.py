"""
HFX3D Dataset Loader for PointNet++ semantic segmentation.

Dataset path: set via the HFX3D_DATA_ROOT env var (see repo README).
  train/  and  test/  subdirs, each with *.h5 files (one per building).

Each .h5 file contains:
  point_cloud/xyz        (N, 3) float32  mean-centred, metric scale preserved
  point_cloud/rgb        (N, 3) float32  [0, 1]
  point_cloud/intensity  (N,)   float32  [0, 1]  (global p1/p99 normalisation)
  labels/semantic        (N,)   uint8    0-11

12 classes:
  wall(0) window(1) door(2) balcony(3) vegetation(4) stairs(5)
  terrain(6) roof(7) blinds(8) other(9) column(10) arch(11)

Feature vector fed to PointNet++ / PointNet (7-D):
  dim 0-2 : xyz centred within the sampled point set
  dim 3-5 : rgb  [0, 1]
  dim 6   : intensity  [0, 1]
All 5 benchmark models use this same 7-D feature set for fair comparison.
"""

import os
import glob
import numpy as np
import h5py
from torch.utils.data import Dataset
from scipy.spatial import cKDTree

NUM_CLASSES = 12
CLASS_NAMES = [
    "wall", "window", "door", "balcony", "vegetation", "stairs",
    "terrain", "roof", "blinds", "other", "column", "arch",
]


class HFX3DDataset(Dataset):
    """
    Random-sampling dataset for PointNet++ training / evaluation.

    Each __getitem__ call picks a random building (weighted by building size),
    then randomly samples `num_point` points from that building.
    """

    def __init__(
        self,
        split="train",
        data_root=os.environ.get("HFX3D_DATA_ROOT", "/project/6118401/abdiaziz/data/HFX3D_Splits"),
        num_point=4096,
        sample_rate=1.0,
        transform=None,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), f"split must be 'train' or 'test', got '{split}'"
        self.num_point = num_point
        self.transform = transform

        split_dir = os.path.join(data_root, split)
        h5_files = sorted(glob.glob(os.path.join(split_dir, "*.h5")))
        if len(h5_files) == 0:
            raise FileNotFoundError(f"No .h5 files found in {split_dir}")

        # Load all buildings into memory
        self.building_points = []   # list of (N, 7): xyz rgb intensity
        self.building_labels = []   # list of (N,)
        num_point_all = []
        labelweights = np.zeros(NUM_CLASSES)

        print(f"Loading {len(h5_files)} buildings for split='{split}' ...")
        for fp in h5_files:
            with h5py.File(fp, "r") as f:
                xyz = f["point_cloud/xyz"][:]            # (N, 3) float32
                rgb = f["point_cloud/rgb"][:]            # (N, 3) float32
                intensity = f["point_cloud/intensity"][:]  # (N,)   float32
                labels = f["labels/semantic"][:]         # (N,)   uint8

            feats = np.concatenate(
                [xyz, rgb, intensity[:, np.newaxis]], axis=1
            ).astype(np.float32)                         # (N, 7)

            tmp, _ = np.histogram(labels, range(NUM_CLASSES + 1))
            labelweights += tmp

            self.building_points.append(feats)
            self.building_labels.append(labels.astype(np.int64))
            num_point_all.append(labels.shape[0])

        # KD-tree per building for spatial (local) crop sampling — see __getitem__.
        self.building_trees = [cKDTree(p[:, :3]) for p in self.building_points]

        # Inverse-frequency label weights (cube-root smoothed, same as S3DIS)
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(
            np.amax(labelweights) / (labelweights + 1e-6), 1.0 / 3.0
        ).astype(np.float32)
        print("Label weights:", self.labelweights)

        # Build index array: each building weighted by its point count
        num_point_all = np.array(num_point_all, dtype=np.float64)
        sample_prob = num_point_all / num_point_all.sum()
        num_iter = int(num_point_all.sum() * sample_rate / num_point)
        building_idxs = []
        for bidx in range(len(h5_files)):
            building_idxs.extend([bidx] * int(round(sample_prob[bidx] * num_iter)))
        self.building_idxs = np.array(building_idxs)
        print(f"Totally {len(self.building_idxs)} samples in '{split}' set.")

    def __len__(self):
        return len(self.building_idxs)

    def __getitem__(self, idx):
        bidx = self.building_idxs[idx]
        feats = self.building_points[bidx]   # (N, 7)
        labels = self.building_labels[bidx]  # (N,)
        N = feats.shape[0]

        # Spatial local crop: random centre + its num_point nearest neighbours.
        # Matches the test-time KNN crops in full_scene_infer.py. The old code
        # sampled num_point points scattered across the whole ~19M-pt building,
        # so PointNet++ ball-query grouping found no real local neighbourhoods
        # and rare classes were almost never present in a sample.
        tree = self.building_trees[bidx]
        k = min(self.num_point, N)
        center_xyz = feats[np.random.randint(N), :3]
        _, nn = tree.query(center_xyz, k=k)
        nn = np.asarray(nn).reshape(-1)
        if k < self.num_point:
            pad = np.random.choice(k, self.num_point - k, replace=True)
            nn = np.concatenate([nn, nn[pad]])
        chosen = nn

        sel_feats = feats[chosen]    # (num_point, 7)
        sel_labels = labels[chosen]  # (num_point,)
        sel_xyz = sel_feats[:, :3]   # (num_point, 3)
        center = sel_xyz.mean(axis=0)

        # 9-channel feature: [centred xyz | rgb | intensity | norm_x | norm_y]
        current_points = np.zeros((self.num_point, 7), dtype=np.float32)
        current_points[:, 0:3] = sel_xyz - center               # centred xyz
        current_points[:, 3:6] = sel_feats[:, 3:6]              # rgb  [0,1]
        current_points[:, 6] = sel_feats[:, 6]                  # intensity

        if self.transform is not None:
            current_points, sel_labels = self.transform(current_points, sel_labels)

        return current_points, sel_labels


if __name__ == "__main__":
    import torch
    ds = HFX3DDataset(split="train", num_point=4096)
    pts, lbl = ds[0]
    print("points shape:", pts.shape)    # (4096, 9)
    print("labels shape:", lbl.shape)    # (4096,)
    print("unique labels:", np.unique(lbl))
    print("label weights:", ds.labelweights)
