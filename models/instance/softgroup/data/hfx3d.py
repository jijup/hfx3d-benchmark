"""HFX3D dataset for SoftGroup.

Replaces the old offline-block pipeline (dataset/hfx3d/prepare_data_hfx3d.py:
non-overlapping 15 m blocks, rgb remapped to [-1,1], 3-channel input). Here:

* Reads the released HFX3D_Instance .h5 files directly (via an uncompressed
  .npy mmap cache, built once by ensure_cache(); the cache is a byte-exact
  copy of the h5 arrays, no transformation).
* Training crops: the same sampler as the PointGroup baseline
  (Pointcept ColumnCrop, mode="random"): centre = xy of a uniformly drawn
  point, keep every point with |dx|,|dy| <= 7.5 m (full height), and if more
  than 250k points remain, keep a random 250k subset. Buildings are visited
  in a shuffled order, each building once per 18 samples (PointGroup's
  loop-over-buildings sampler). Every sample i uses its own RandomState seeded
  from (seed, i), so the data stream is identical regardless of worker count
  and is exactly resumable.
* Input: 7-D [x, y, z, r, g, b, intensity] as released (xyz in metres,
  rgb in [0,1], intensity in [0,1]). feats = [r, g, b, intensity]; the model
  appends coords_float (xyz, metres) because with_coords=True -> 7 channels.
* Labels: raw ids are remapped to SoftGroup's stuff-first dense order
  (stuff = wall, terrain, roof, other -> 0..3; things -> 4..11). Instance
  targets: stuff points AND thing points with instance == -1 (ignored <200-pt
  GT objects) get -100, i.e. they are excluded from offset/instance targets.
"""
import glob
import math
import os
import os.path as osp
from multiprocessing import Pool

import h5py
import numpy as np
import scipy.interpolate
import scipy.ndimage
import torch

from .custom import CustomDataset

# raw HFX3D id -> SoftGroup dense id (stuff-first). Same table as the old
# prepare_data_hfx3d.py RAW_TO_DENSE (unchanged).
RAW_TO_DENSE = np.array([0, 4, 5, 6, 7, 8, 1, 2, 9, 3, 10, 11], dtype=np.int64)
DENSE_TO_RAW = np.argsort(RAW_TO_DENSE)  # dense id -> raw id
LABEL_SHIFT = 4  # semantic_classes - instance_classes
# instance-class index k (0..7, SoftGroup label_id = k+1) -> raw HFX3D id
THING_RAW = DENSE_TO_RAW[LABEL_SHIFT:]  # [1, 2, 3, 4, 5, 8, 10, 11]
RAW_CLASS_NAMES = ("wall", "window", "door", "balcony", "vegetation", "stairs", "terrain", "roof",
                   "blinds", "other", "column", "arch")
CACHE_KEYS = ("xyz", "feat", "sem", "inst")


# ----------------------------------------------------------------------------- cache
def _cache_one(args):
    h5_path, out_dir = args
    if all(osp.isfile(osp.join(out_dir, f"{k}.npy")) for k in CACHE_KEYS):
        return out_dir
    tmp = out_dir + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        xyz = f["point_cloud/xyz"][:].astype(np.float32)
        rgb = f["point_cloud/rgb"][:].astype(np.float32)
        inten = f["point_cloud/intensity"][:].astype(np.float32).reshape(-1, 1)
        sem = f["labels/semantic"][:].reshape(-1).astype(np.uint8)
        inst = f["labels/instance"][:].reshape(-1).astype(np.int32)
    np.save(osp.join(tmp, "xyz.npy"), xyz)
    np.save(osp.join(tmp, "feat.npy"), np.concatenate([rgb, inten], 1))  # (N,4) rgb|intensity
    np.save(osp.join(tmp, "sem.npy"), sem)  # RAW semantic ids 0..11
    np.save(osp.join(tmp, "inst.npy"), inst)  # RAW instance ids (-1 = ignore)
    if osp.isdir(out_dir):  # stale partial dir
        for k in CACHE_KEYS:
            if osp.isfile(osp.join(out_dir, f"{k}.npy")):
                os.remove(osp.join(out_dir, f"{k}.npy"))
        os.rmdir(out_dir)
    os.rename(tmp, out_dir)
    return out_dir


def ensure_cache(data_root, split, cache_root, names=None, processes=16):
    """Uncompressed .npy copy of each building's h5 arrays (mmap-able, shared
    page cache across DataLoader workers). Returns sorted [(name, dir)]."""
    paths = sorted(glob.glob(osp.join(data_root, split, "*.h5")))
    assert paths, f"no h5 files in {osp.join(data_root, split)}"
    if names:
        paths = [p for p in paths if osp.splitext(osp.basename(p))[0] in names]
    jobs = [(p, osp.join(cache_root, split, osp.splitext(osp.basename(p))[0])) for p in paths]
    todo = [j for j in jobs if not all(osp.isfile(osp.join(j[1], f"{k}.npy")) for k in CACHE_KEYS)]
    if todo:
        os.makedirs(osp.join(cache_root, split), exist_ok=True)
        with Pool(min(processes, len(todo))) as pool:
            pool.map(_cache_one, todo)
    return [(osp.basename(d), d) for _, d in jobs]


def load_building(cache_dir, mmap=True):
    mode = "r" if mmap else None
    return {k: np.load(osp.join(cache_dir, f"{k}.npy"), mmap_mode=mode) for k in CACHE_KEYS}


def split_building_names(data_root, split):
    return [osp.splitext(osp.basename(p))[0]
            for p in sorted(glob.glob(osp.join(data_root, split, "*.h5")))]


# ----------------------------------------------------------------------------- helpers
def sample_rng(seed, index):
    return np.random.RandomState(np.random.SeedSequence([seed, index]).generate_state(4))


def column_crop(xy, rng, footprint=15.0, point_max=250000):
    """Exact replica of Pointcept ColumnCrop(mode='random') used by PointGroup,
    with an explicit RandomState instead of the global np.random."""
    half = footprint / 2
    center_xy = xy[rng.randint(xy.shape[0])]
    idx = np.flatnonzero((np.abs(xy[:, 0] - center_xy[0]) <= half)
                         & (np.abs(xy[:, 1] - center_xy[1]) <= half))
    if idx.shape[0] > point_max:
        idx = rng.choice(idx, point_max, replace=False)
    return idx


def center_xy(xyz):
    """Crop/tile-local frame: xy centred on the crop's xy bounding-box centre,
    z kept as released (building-mean-centred metres). Used identically for
    training crops and evaluation tiles."""
    xyz = np.array(xyz, dtype=np.float32, copy=True)
    xyz[:, :2] -= (xyz[:, :2].min(0) + xyz[:, :2].max(0)) / 2
    return xyz


def dense_semantic(raw_sem):
    return RAW_TO_DENSE[raw_sem.astype(np.int64)]


def instance_targets(raw_inst, sem_dense):
    """0..K-1 contiguous instance ids for valid thing instances; -100 for
    stuff points and for ignored (-1, <200-pt) thing instances."""
    valid = (sem_dense >= LABEL_SHIFT) & (raw_inst != -1)
    out = np.full(raw_inst.shape[0], -100, dtype=np.int64)
    if valid.any():
        _, inv = np.unique(raw_inst[valid], return_inverse=True)
        out[valid] = inv
    return out


def instance_info(xyz_middle, inst, sem_dense):
    """Vectorised equivalent of CustomDataset.getInstanceInfo + HFX3D's
    instance_cls shift (dense thing id 4..11 -> 0..7)."""
    n_inst = int(inst.max()) + 1 if (inst >= 0).any() else 0
    offset = np.zeros_like(xyz_middle, dtype=np.float32)  # masked out for -100 points
    if n_inst == 0:
        return 0, [], [], offset
    v = inst >= 0
    counts = np.bincount(inst[v], minlength=n_inst)
    sums = np.stack([np.bincount(inst[v], weights=xyz_middle[v, d], minlength=n_inst)
                     for d in range(3)], 1)
    mean = sums / np.maximum(counts, 1)[:, None]
    offset[v] = (mean[inst[v]] - xyz_middle[v]).astype(np.float32)
    idx_v = np.flatnonzero(v)
    _, first_in_v = np.unique(inst[v], return_index=True)  # ids are contiguous 0..n_inst-1
    first = idx_v[first_in_v]  # first point of each instance (same as upstream's [0])
    cls = (sem_dense[first] - LABEL_SHIFT).tolist()
    return n_inst, counts.tolist(), cls, offset


def elastic(x, gran, mag, rng):
    """Upstream CustomDataset.elastic with an explicit RNG and integer grid size
    (gran may be fractional after rescaling)."""
    blur0 = np.ones((3, 1, 1)).astype('float32') / 3
    blur1 = np.ones((1, 3, 1)).astype('float32') / 3
    blur2 = np.ones((1, 1, 3)).astype('float32') / 3
    bb = (np.abs(x).max(0) // gran).astype(np.int32) + 3
    noise = [rng.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
    for blur in (blur0, blur1, blur2, blur0, blur1, blur2):
        noise = [scipy.ndimage.convolve(n, blur, mode='constant', cval=0) for n in noise]
    ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
    interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0)
              for n in noise]
    return x + np.hstack([i(x)[:, None] for i in interp]) * mag


def augment_xyz(xyz, rng):
    """Upstream CustomDataset.dataAugment(jitter, flip, rot, scale) with an
    explicit RNG (same distributions)."""
    m = np.eye(3)
    m += rng.randn(3, 3) * 0.1
    m[0][0] *= rng.randint(0, 2) * 2 - 1
    theta = rng.rand() * 2 * math.pi
    m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                      [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    xyz = xyz * rng.uniform(0.95, 1.05)
    return np.matmul(xyz, m).astype(np.float32)


def make_sample(scan_id, xyz_middle, feat, sem_dense, inst, scale, xyz_vox=None):
    """Pack one crop/tile into SoftGroup's per-sample tuple (CustomDataset.__getitem__)."""
    if xyz_vox is None:
        xyz_vox = xyz_middle * scale
    xyz_vox = xyz_vox - xyz_vox.min(0)
    n_inst, inst_pointnum, inst_cls, pt_offset = instance_info(xyz_middle, inst, sem_dense)
    return (scan_id, torch.from_numpy(xyz_vox).long(), torch.from_numpy(xyz_middle).float(),
            torch.from_numpy(np.ascontiguousarray(feat)).float(), torch.from_numpy(sem_dense),
            torch.from_numpy(inst), n_inst, inst_pointnum, inst_cls, torch.from_numpy(pt_offset))


# ----------------------------------------------------------------------------- dataset
class HFX3DDataset(CustomDataset):
    """Iteration-indexed training stream of random 15 m column crops.

    Sample k of this dataset is GLOBAL sample index (start_sample + k); the
    trainer sets start_sample = global_step * batch_size, so the crops seen
    at any optimizer step are fixed by (seed, step) alone.
    """

    CLASSES = ("window", "door", "balcony", "vegetation", "stairs", "blinds", "column", "arch")

    def __init__(self, data_root, prefix, cache_root, voxel_cfg=None, training=True, logger=None,
                 seed=42, footprint=15.0, aug_prob=1.0, color_noise_std=0.05,
                 elastic_ref_scale=50.0, num_samples=0, start_sample=0, names=None, **kwargs):
        assert training, 'HFX3DDataset is a training stream; evaluation uses ' \
            'tools/test_fullscene_softgroup_hfx3d.py'
        self.data_root, self.prefix, self.cache_root = data_root, prefix, cache_root
        self.voxel_cfg, self.training, self.logger = voxel_cfg, training, logger
        self.mode = 'train'
        self.seed, self.footprint, self.aug_prob = seed, footprint, aug_prob
        self.color_noise_std = color_noise_std
        # Upstream elastic params (6,40) and (20,160) are in VOXEL units and were
        # designed at scale 50 (2 cm voxels, ScanNet/S3DIS). Rescale so the
        # distortion has the same METRIC granularity/magnitude at our scale.
        self.elastic_factor = voxel_cfg.scale / elastic_ref_scale
        self.buildings = ensure_cache(data_root, prefix, cache_root, names=names)
        self.data = [load_building(d) for _, d in self.buildings]
        self.names = [n for n, _ in self.buildings]
        self.num_samples, self.start_sample = num_samples, start_sample
        if logger is not None:
            logger.info(f'HFX3D {prefix}: {len(self.names)} buildings, '
                        f'{sum(d["xyz"].shape[0] for d in self.data)} points, '
                        f'elastic factor {self.elastic_factor:.3f}')

    def set_range(self, start_sample, num_samples):
        self.start_sample, self.num_samples = start_sample, num_samples

    def __len__(self):
        return self.num_samples

    def building_for(self, s):
        nb = len(self.data)
        order = np.random.RandomState(
            np.random.SeedSequence([self.seed, 1, s // nb]).generate_state(4)).permutation(nb)
        return int(order[s % nb])

    def __getitem__(self, k):
        s = self.start_sample + k
        rng = sample_rng(self.seed, s)
        b = self.building_for(s)
        d = self.data[b]
        xy = d['xyz'][:, :2]
        for _ in range(50):  # resample centre if the column is too sparse (keeps batch exactly 4)
            idx = column_crop(xy, rng, self.footprint, self.voxel_cfg.max_npoint)
            if idx.shape[0] >= self.voxel_cfg.min_npoint:
                break
        idx = np.sort(idx)
        xyz = center_xy(d['xyz'][idx])
        feat = np.array(d['feat'][idx], dtype=np.float32)
        sem_dense = dense_semantic(d['sem'][idx])
        inst = instance_targets(np.asarray(d['inst'][idx]), sem_dense)

        scale = self.voxel_cfg.scale
        xyz_middle = augment_xyz(xyz, rng)
        xyz_vox = xyz_middle * scale
        if rng.rand() < self.aug_prob:
            f = self.elastic_factor
            xyz_vox = elastic(xyz_vox, 6 * f, 40 * f, rng)
            xyz_vox = elastic(xyz_vox, 20 * f, 160 * f, rng)
        # Upstream adds N(0, 0.1) per-channel to colour in [-1,1]; HFX3D colour is
        # [0,1] (half the range) -> 0.05. Applied to rgb only, not intensity.
        feat[:, :3] += (rng.randn(3) * self.color_noise_std).astype(np.float32)
        return make_sample(f'{self.names[b]}_s{s}', xyz_middle, feat, sem_dense, inst, scale,
                           xyz_vox=xyz_vox)
