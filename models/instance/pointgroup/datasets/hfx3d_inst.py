"""
HFX3D Instance Segmentation Dataset

Urban building facade point cloud dataset for panoptic/instance segmentation.
12 semantic classes (8 "thing" + 4 "stuff"); see the paper.

A Pointcept dataset (https://github.com/Pointcept/Pointcept), registered as
HFX3DInstDataset. Author: Abdiaziz.
"""

import os
import numpy as np
import h5py

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class HFX3DInstDataset(DefaultDataset):
    """
    HFX3D dataset for urban building facade instance/panoptic segmentation.
    Loads HDF5 files directly (one file per building); no offline conversion
    step required.

    Dataset structure:
        data_root/
        ├── train/*.h5
        ├── val/*.h5
        └── test/*.h5

    Each HDF5 file contains:
        - point_cloud/xyz: (N, 3) float32, mean-centred, metric scale (NOT
          unit-sphere normalized -- this differs from the semantic-only
          HFX3D_Splits dataset used by HFX3DDataset)
        - point_cloud/rgb: (N, 3) float32 in [0, 1]
        - point_cloud/intensity: (N) float32 in [0, 1]
        - labels/semantic: (N) uint8, classes 0-11
        - labels/instance: (N) int32, panoptic ids 1..M ('-1' = ignore, no '0')
        - labels/superpoint: (N) int32, ids 0..S-1 (not used by PointGroup;
          not loaded here -- superpoint transformer models read the .h5 via
          their own converters)

    Classes (12): wall(0) window(1) door(2) balcony(3) vegetation(4) stairs(5)
    terrain(6) roof(7) blinds(8) other(9) column(10) arch(11)

    "Thing" classes scored with instance AP (8): window, door, balcony,
    vegetation, stairs, blinds, column, arch.
    "Stuff" classes excluded from instance AP (4): wall, terrain, roof, other.
    Pass segment_ignore_index=(-1, 0, 6, 7, 9) to the PointGroup model config
    to encode this split (see configs/hfx3d/insseg-pointgroup-v1m1-0-spunet-base.py).
    """

    class_names = [
        "wall", "window", "door", "balcony", "vegetation", "stairs",
        "terrain", "roof", "blinds", "other", "column", "arch",
    ]
    num_classes = 12
    class2id = np.arange(12)

    def get_data(self, idx):
        hdf5_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)

        # In-process cache, keyed by building name -- NOT the same mechanism
        # as the inherited `self.cache`/shared_dict path (removed below).
        # That path calls shared_dict(cache_name) with no `var=`, which only
        # ATTACHES to an already-populated /dev/shm entry -- it does not
        # create one, so it would crash unless a separate pre-caching step
        # populated it first (no such step exists in this project). Root
        # cause of a real bug: our PointGroup config uses `loop=20` (protocol
        # training-tile density -- see insseg-pointgroup config) so each
        # of the 18 physical building H5 files gets read in full 20x per
        # epoch; the training log showed periodic 100-300s per-iteration
        # stalls (`Data 234.182 (29.287)`)
        # between otherwise-~0.2s iterations -- repeated full re-reads of
        # multi-million-point HDF5 arrays over Lustre. This plain dict cache
        # (scoped to this dataset instance/process) fixes that: each building
        # is read from disk once per worker process, not once per access.
        if not hasattr(self, "_hfx3d_cache"):
            self._hfx3d_cache = {}
        if name in self._hfx3d_cache:
            return {
                k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in self._hfx3d_cache[name].items()
            }

        with h5py.File(hdf5_path, "r") as f:
            xyz = f["point_cloud/xyz"][:].astype(np.float32)
            rgb = f["point_cloud/rgb"][:].astype(np.float32)
            intensity = f["point_cloud/intensity"][:].astype(np.float32)
            segment = f["labels/semantic"][:].reshape([-1]).astype(np.int32)
            instance = f["labels/instance"][:].reshape([-1]).astype(np.int32)

        # dummy z-up normals -- HFX3D has no precomputed normals; matches
        # HFX3DDataset's convention for the semantic track
        normal = np.tile([0, 0, 1], (len(xyz), 1)).astype(np.float32)

        data_dict = {
            "coord": xyz,
            # HDF5 rgb is [0, 1]; Pointcept colour transforms and
            # NormalizeColor assume 0-255, so scale here (NormalizeColor maps back to [0, 1]).
            "color": rgb * 255.0,
            "normal": normal,
            "intensity": intensity.reshape(-1, 1),
            "segment": segment,
            "instance": instance,
            "name": name,
            "index_valid_keys": [
                "coord",
                "color",
                "normal",
                "intensity",
                "segment",
                "instance",
            ],
        }
        self._hfx3d_cache[name] = data_dict
        return {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in data_dict.items()
        }

    def get_data_name(self, idx):
        hdf5_path = self.data_list[idx % len(self.data_list)]
        return os.path.splitext(os.path.basename(hdf5_path))[0]
