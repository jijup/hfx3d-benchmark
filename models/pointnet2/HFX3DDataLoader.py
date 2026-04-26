"""
HFX3D Dataset Loader for PointNet++ semantic segmentation.

Dataset path: /home/abdiaziz/projects/def-jijup/abdiaziz/WACV_Splits/
  train/  and  test/  subdirs, each with *.h5 files (one per building).

Each .h5 file contains:
  point_cloud/xyz        (N, 3) float32  unit-sphere normalised
  point_cloud/rgb        (N, 3) float32  [0, 1]
  point_cloud/intensity  (N,)   float32  [0, 1]
  labels/semantic        (N,)   uint8    0-11

12 classes:
  wall(0) window(1) door(2) balcony(3) vegetation(4) stairs(5)
  terrain(6) roof(7) blinds(8) other(9) column(10) arch(11)

Feature vector fed to PointNet++ (9-D, matching pointnet2_sem_seg.py):
  dim 0-2 : xyz centred within the sampled point set
  dim 3-5 : rgb  [0, 1]
  dim 6   : intensity  [0, 1]
  dim 7   : (x - x_min) / x_range   (global normalised x)
  dim 8   : (y - y_min) / y_range   (global normalised y)

This matches the 9-channel convention used by the original S3DIS loader,
allowing pointnet2_sem_seg.py (which sets sa1 in_channel = 9+3) to be
used without any model modification.
"""

import os
import glob
import numpy as np
import h5py
from torch.utils.data import Dataset

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
        data_root="/home/abdiaziz/projects/def-jijup/abdiaziz/WACV_Splits",
        num_point=4096,
        sample_rate=1.0,
        transform=None,
    ):
        super().__init__()
        assert split in ("train", "test"), f"split must be 'train' or 'test', got '{split}'"
        self.num_point = num_point
        self.transform = transform

        split_dir = os.path.join(data_root, split)
        h5_files = sorted(glob.glob(os.path.join(split_dir, "*.h5")))
        if len(h5_files) == 0:
            raise FileNotFoundError(f"No .h5 files found in {split_dir}")

        # Load all buildings into memory
        self.building_points = []   # list of (N, 7): xyz rgb intensity
        self.building_labels = []   # list of (N,)
        self.coord_min = []
        self.coord_max = []
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
            self.coord_min.append(xyz.min(axis=0))
            self.coord_max.append(xyz.max(axis=0))
            num_point_all.append(labels.shape[0])

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

        cmin = self.coord_min[bidx]
        cmax = self.coord_max[bidx]
        crange = (cmax - cmin) + 1e-6

        # Random sample
        replace = N < self.num_point
        chosen = np.random.choice(N, self.num_point, replace=replace)

        sel_feats = feats[chosen]    # (num_point, 7)
        sel_labels = labels[chosen]  # (num_point,)
        sel_xyz = sel_feats[:, :3]   # (num_point, 3)
        center = sel_xyz.mean(axis=0)

        # 9-channel feature: [centred xyz | rgb | intensity | norm_x | norm_y]
        current_points = np.zeros((self.num_point, 9), dtype=np.float32)
        current_points[:, 0:3] = sel_xyz - center               # centred xyz
        current_points[:, 3:6] = sel_feats[:, 3:6]              # rgb  [0,1]
        current_points[:, 6] = sel_feats[:, 6]                  # intensity
        current_points[:, 7] = (sel_xyz[:, 0] - cmin[0]) / crange[0]  # norm x
        current_points[:, 8] = (sel_xyz[:, 1] - cmin[1]) / crange[1]  # norm y

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
