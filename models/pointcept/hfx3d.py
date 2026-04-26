"""
HFX3D Dataset

Urban building facades point cloud dataset for semantic segmentation.
12 classes: wall, window, door, balcony, vegetation, stairs, terrain, roof, blinds, other, column, arch

Author: Abdiaziz
"""

import os
import numpy as np
import h5py
from scipy.spatial import cKDTree
from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class HFX3DDataset(DefaultDataset):
    """
    HFX3D dataset for urban building facade semantic segmentation.
    Loads HDF5 files directly and samples points dynamically during training.

    Dataset structure:
        data_root/
        ├── train/
        │   ├── building001.hdf5
        │   ├── building002.hdf5
        │   └── ...
        └── test/
            ├── building101.hdf5
            └── ...

    Each HDF5 file contains:
        - point_cloud/xyz: (N, 3) coordinates
        - point_cloud/rgb: (N, 3) RGB values [0, 1]
        - point_cloud/intensity: (N,) LiDAR intensity
        - labels/semantic: (N,) semantic labels [0-11]

    Classes (12):
        0: wall, 1: window, 2: door, 3: balcony, 4: vegetation, 5: stairs,
        6: terrain, 7: roof, 8: blinds, 9: other, 10: column, 11: arch
    """

    # Class names in order (0-11)
    class_names = [
        "wall", "window", "door", "balcony", "vegetation", "stairs",
        "terrain", "roof", "blinds", "other", "column", "arch"
    ]

    # Number of classes
    num_classes = 12

    # Labels are already 0-indexed (0-11), so identity mapping
    class2id = np.arange(12)

    def __init__(self,
                 max_points_per_building=500000,
                 sample_method='random',
                 compute_normals=True,
                 **kwargs):
        """
        Args:
            max_points_per_building: Maximum points to load per building (for memory efficiency)
            sample_method: 'random' or 'fps' (farthest point sampling)
            compute_normals: Whether to compute normals on-the-fly
        """
        self.max_points = max_points_per_building
        self.sample_method = sample_method
        self.compute_normals_flag = compute_normals
        super().__init__(**kwargs)

    def compute_normals(self, xyz, k=20):
        """
        Compute point cloud normals using PCA on k-nearest neighbors.

        Args:
            xyz: (N, 3) array of coordinates
            k: number of neighbors for normal estimation

        Returns:
            normals: (N, 3) array of unit normals
        """
        tree = cKDTree(xyz)
        normals = np.zeros_like(xyz, dtype=np.float32)

        # Process in batches to avoid memory issues
        batch_size = 10000
        for i in range(0, len(xyz), batch_size):
            end_idx = min(i + batch_size, len(xyz))
            batch_xyz = xyz[i:end_idx]

            # Find k nearest neighbors
            _, indices = tree.query(batch_xyz, k=min(k, len(xyz)))

            # Compute normal for each point
            for j, neighbor_ids in enumerate(indices):
                neighbors = xyz[neighbor_ids]
                # Center the neighbors
                centered = neighbors - neighbors.mean(axis=0)
                # PCA: smallest eigenvector is the normal
                try:
                    cov = np.cov(centered.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    normal = eigenvectors[:, 0]  # Smallest eigenvalue
                    normals[i + j] = normal / (np.linalg.norm(normal) + 1e-8)
                except:
                    normals[i + j] = [0, 0, 1]  # Default to z-up

        return normals.astype(np.float32)

    def get_data(self, idx):
        """
        Load data from HDF5 file with optional sampling.

        Args:
            idx: Building index

        Returns:
            data_dict: Dictionary with keys:
                - coord: (N, 3) float32 coordinates (centered)
                - color: (N, 3) float32 RGB normalized to [0, 1]
                - normal: (N, 3) float32 surface normals
                - segment: (N,) int32 semantic labels (0-11)
                - instance: (N,) int32 instance IDs (all -1 for HFX3D)
                - name: str building name
        """
        hdf5_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)

        if self.cache:
            cache_name = f"hfx3d-{name}"
            from pointcept.utils.cache import shared_dict
            return shared_dict(cache_name)

        # Load from HDF5
        try:
            with h5py.File(hdf5_path, 'r') as f:
                xyz = f['point_cloud/xyz'][:]
                rgb = f['point_cloud/rgb'][:]
                intensity = f['point_cloud/intensity'][:]

                # Try to load precomputed normals if available
                if 'point_cloud/normals' in f:
                    normals = f['point_cloud/normals'][:]
                else:
                    normals = None  # Will use dummy normals later

                if 'labels/semantic' in f:
                    labels = f['labels/semantic'][:].astype(np.int32)
                else:
                    labels = np.full(len(xyz), -1, dtype=np.int32)
        except Exception as e:
            raise RuntimeError(f"Error loading {hdf5_path}: {str(e)}")

        num_points = len(xyz)

        # Sample points if building is too large
        if num_points > self.max_points:
            if self.sample_method == 'random':
                # Random sampling
                indices = np.random.choice(num_points, self.max_points, replace=False)
            else:
                # For now, fall back to random (FPS is slower)
                indices = np.random.choice(num_points, self.max_points, replace=False)

            xyz = xyz[indices]
            rgb = rgb[indices]
            intensity = intensity[indices]
            labels = labels[indices]
            if normals is not None:
                normals = normals[indices]

        # Data is already normalized from LAS converter:
        # - XYZ: centered and unit-sphere normalized (within [-1,1])
        # - RGB: divided by 65535.0 (already [0,1])
        # - Intensity: min-max normalized per file (already [0,1])
        # Just ensure correct data types
        xyz = xyz.astype(np.float32)
        rgb = rgb.astype(np.float32)
        intensity = intensity.astype(np.float32)

        # Handle normals: use precomputed if available, otherwise use dummy
        if normals is None:
            # Use dummy z-up normals (fast baseline)
            normals = np.tile([0, 0, 1], (len(xyz), 1)).astype(np.float32)
        else:
            normals = normals.astype(np.float32)

        # Validate labels
        labels = labels.reshape([-1]).astype(np.int32)
        unique_labels = np.unique(labels)
        invalid_labels = unique_labels[(unique_labels >= self.num_classes) & (unique_labels != -1)]

        if len(invalid_labels) > 0:
            print(f"WARNING [{name}]: Found invalid labels {invalid_labels}, mapping to -1")
            labels = np.where((labels >= self.num_classes) & (labels != -1), -1, labels)

        invalid_neg = unique_labels[unique_labels < -1]
        if len(invalid_neg) > 0:
            print(f"WARNING [{name}]: Found invalid negative labels {invalid_neg}, mapping to -1")
            labels = np.where(labels < -1, -1, labels)

        # Create data dictionary
        # Reshape intensity to (N, 1) for concatenation with other features
        intensity_reshaped = intensity.reshape(-1, 1).astype(np.float32)

        data_dict = {
            "coord": xyz,
            "color": rgb,
            "normal": normals,
            "segment": labels,
            "instance": np.full(len(xyz), -1, dtype=np.int32),  # No instance labels
            "intensity": intensity_reshaped,  # (N, 1) shape for feature concatenation
            "name": name,
            # Tell transforms to apply indexing to intensity
            "index_valid_keys": [
                "coord",
                "color",
                "normal",
                "intensity",  # Add intensity here!
                "segment",
                "instance",
            ],
        }

        return data_dict

    def get_data_name(self, idx):
        """Get building name from HDF5 file path."""
        hdf5_path = self.data_list[idx % len(self.data_list)]
        return os.path.splitext(os.path.basename(hdf5_path))[0]


@DATASETS.register_module()
class HFX3DDatasetWithIntensity(HFX3DDataset):
    """
    HFX3D dataset variant that includes LiDAR intensity as an additional feature.

    This increases input channels from 9 (xyz + rgb + normal) to 10 (xyz + rgb + normal + intensity).
    Use this if you want to leverage intensity information in your model.
    """

    def get_data(self, idx):
        """
        Load data with intensity channel (already included in parent class).
        """
        data_dict = super().get_data(idx)

        # Ensure intensity is present
        if "intensity" not in data_dict or data_dict["intensity"] is None:
            # Use zero intensity if not available
            data_dict["intensity"] = np.zeros(
                data_dict["coord"].shape[0], dtype=np.float32
            )

        return data_dict
