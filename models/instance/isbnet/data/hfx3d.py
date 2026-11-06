import numpy as np

from .custom import CustomDataset


def tile_local_coords(xyz):
    """Shift a crop/tile to its own local frame: XY bounding-box midpoint at
    0, lowest point at z = 0. Same convention as Pointcept's
    CenterShift(apply_z=True), which the PointGroup baseline applies to
    every training crop and every evaluation tile, so the xyz feature that
    both models see (with_coords=True) is tile-local in the same way.
    Pure translation: metric scale is untouched."""
    xyz_min = xyz.min(0)
    xyz_max = xyz.max(0)
    shift = np.array(
        [(xyz_min[0] + xyz_max[0]) / 2.0, (xyz_min[1] + xyz_max[1]) / 2.0, xyz_min[2]], dtype=xyz.dtype
    )
    return xyz - shift


class HFX3DDataset(CustomDataset):
    """
    HFX3D facade instance-seg benchmark.
    Modeled on STPLS3DDataset (ISBNet's closest outdoor precedent).

    Data: one directory of memory-mapped .npy arrays per split, written by
    isbnet_hfx3d_converter.py:
        <name>_xyz.npy  float32 (N,3)  metres, building-mean-centred (as released)
        <name>_feat.npy float32 (N,4)  [r, g, b, intensity], all in [0,1] (as released)
        <name>_sem.npy  int64   (N)   dense stuff-first ids (wall/terrain/roof/other = 0-3)
        <name>_inst.npy int64   (N)   0..K-1 thing instances, -100 = stuff OR GT instance -1
    The file list globs "*_xyz.npy" (config suffix), one entry per building.

    Training sample: a random 15 m x 15 m full-height column
    centred on a random point (exactly Pointcept's ColumnCrop, used by the
    PointGroup baseline), capped at 250k points by a seeded random subsample
    (np.random is seeded per worker in build_dataloader). The column is drawn
    BEFORE any augmentation so the expensive elastic distortion only ever runs
    on the crop.
    """

    CLASSES = (
        "wall", "terrain", "roof", "other",
        "window", "door", "balcony", "vegetation", "stairs", "blinds",
        "column", "arch",
    )

    COLUMN_FOOTPRINT = 15.0   # metres
    POINT_MAX = 250000        # forward-pass cap
    MAX_CROP_TRIES = 10       # redraw a column that falls below voxel_cfg.min_npoint
    # ISBNet's train-time colour noise is randn*0.1 on colour in [-1,1]; the
    # same relative perturbation on the released [0,1] colour is 0.05. It is
    # applied to the 3 colour channels only (intensity is not colour).
    COLOR_CHANNELS = 3
    COLOR_NOISE_STD = 0.05

    def load(self, filename):
        base = filename[: -len("_xyz.npy")]
        xyz_all = np.load(base + "_xyz.npy", mmap_mode="r")
        feat_all = np.load(base + "_feat.npy", mmap_mode="r")
        sem_all = np.load(base + "_sem.npy", mmap_mode="r")
        inst_all = np.load(base + "_inst.npy", mmap_mode="r")

        if self.training:
            half = self.COLUMN_FOOTPRINT / 2.0
            xy = xyz_all[:, :2]
            for _ in range(self.MAX_CROP_TRIES):
                center_xy = np.asarray(xy[np.random.randint(xy.shape[0])])
                idx = np.flatnonzero(np.all(np.abs(xy - center_xy) <= half, axis=1))
                if idx.shape[0] >= self.voxel_cfg.min_npoint:
                    break
            if idx.shape[0] > self.POINT_MAX:
                idx = np.sort(np.random.choice(idx, self.POINT_MAX, replace=False))
        else:
            idx = np.arange(xyz_all.shape[0])

        xyz = np.asarray(xyz_all[idx], dtype=np.float32)
        rgb = np.asarray(feat_all[idx], dtype=np.float32)
        semantic_label = np.asarray(sem_all[idx], dtype=np.int64)
        instance_label = np.asarray(inst_all[idx], dtype=np.int64)
        if self.training:
            xyz = tile_local_coords(xyz)

        # No superpoints (use_spp_pool=False, as in ISBNet's STPLS3D config):
        # dummy per-point ids, same as STPLS3DDataset.
        spp = np.arange(xyz.shape[0], dtype=np.int64)
        return xyz, rgb, semantic_label, instance_label, spp

    def getCroppedInstLabel(self, instance_label, valid_idxs):
        # Equivalent to CustomDataset's version (ids made contiguous 0..K-1,
        # -100 kept) but vectorised: the original loops over every id up to
        # the building's max id (~800) with a full scan per id.
        instance_label = instance_label[valid_idxs].copy()
        m = instance_label != -100
        if m.any():
            instance_label[m] = np.unique(instance_label[m], return_inverse=True)[1]
        return instance_label

    def transform_train(self, xyz, rgb, semantic_label, instance_label, spp, aug_prob=1.0):
        # ISBNet's own augmentation (jitter/shear 0.1, flip, full z-rotation),
        # applied to the tile-local metric coordinates.
        xyz_middle = self.dataAugment(xyz, True, True, True, aug_prob)
        xyz = xyz_middle * self.voxel_cfg.scale
        # Shift to the positive octant BEFORE elastic distortion: elastic()
        # sizes its noise grid by max|x|, so on building-centred coordinates
        # it allocated a grid spanning the whole building for every crop.
        xyz = xyz - xyz.min(0)
        if np.random.rand() < aug_prob:
            # Elastic parameters are in VOXEL units. Scale them with the voxel
            # size exactly as SoftGroup's CustomDataset does
            # (6*scale//50, 40*scale/50, ...), i.e. keep the metric meaning
            # ISBNet/SoftGroup use on ScanNet (2 cm voxels): ~0.1 m and
            # ~0.4 m distortion granularity. The old run used the raw
            # STPLS3D numbers at 0.33 m voxels (2 m / 6.7 m granularity).
            s = self.voxel_cfg.scale / 50.0
            xyz = self.elastic(xyz, max(int(6 * s), 1), 40.0 * s)
            xyz = self.elastic(xyz, max(int(20 * s), 1), 160.0 * s)
        xyz = xyz - xyz.min(0)
        # No CustomDataset.crop(): the column crop and 250k cap were already
        # applied in load(), so crop() would be a no-op (and its 512-voxel
        # window logic assumes STPLS3D's 0.33 m voxels).
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, semantic_label, instance_label, spp

    def transform_test(self, xyz, rgb, semantic_label, instance_label, spp):
        # Unchanged ISBNet test transform (including its fixed 0.35*pi
        # rotation). Callers pass tile-local coordinates (tile_local_coords).
        return super().transform_test(xyz, rgb, semantic_label, instance_label, spp)

