"""
HFX3D -> ISBNet data converter.

Notes:
  * colour is written AS RELEASED ([0,1]), not remapped to
    [-1,1]; feat = [r, g, b, intensity], all in [0,1].
  * output is one directory of memory-mappable .npy arrays per split
    (<name>_xyz.npy / _feat.npy / _sem.npy / _inst.npy) instead of a pickled
    .pth tuple, so the 16 data-loader workers share ONE page-cached copy of
    each building instead of re-unpickling a 1-2 GB file per training crop.
  * --splits selects which splits to convert (training only needs train;
    the full-scene evaluator reads the released .h5 files directly).
  Labels, the stuff-first remap, -1/stuff -> -100 ignore and the dense
  instance renumbering are unchanged.

ISBNet (VinAIResearch/ISBNet, CVPR'23) was chosen as HAIS's substitute in the
HFX3D instance-seg benchmark: HAIS's own reference repo is a dead spconv1.0/
CUDA9.2 stack that cannot target either H100 or L40S (compiler predates both
architectures), and a native HAIS reimplementation was scoped as too large
(needs a custom `hierarchical_aggregation` CUDA kernel plus a whole
ScoreNet/mask-refinement subsystem Pointcept's own PointGroup doesn't have).
ISBNet is spconv2-based (same family already proven on this cluster via
Pointcept's native PointGroup and thangvubk/SoftGroup), superpoint-free by
design (its own official STPLS3D config sets use_spp_pool=False -- directly
relevant since 3 superpoint-transformer models were already dropped from
this benchmark for a measured scalability mismatch), and has no
MinkowskiEngine-class dependency risk.

Modeled directly on ISBNet's own STPLS3D preprocessing
(dataset/stpls3d/prepare_data_inst_instance_stpls3d.py, fetched from
VinAIResearch/ISBNet) -- STPLS3D is ISBNet's own closest outdoor-scene
precedent, matching this project's existing "cite the closest domain
precedent" pattern for every other model's dataset adapter.

Output format, confirmed against isbnet/data/custom.py::CustomDataset and
isbnet/data/stpls3d.py::STPLS3DDataset.load() (the class our own
isbnet/data/hfx3d.py subclasses) -- each <out_root>/<split>/<building>.pth
contains a plain tuple, loaded via torch.load():
    (coords, colors, sem_labels, instance_labels)

  coords         : float32 (N, 3), building-mean-centred (HFX3D's raw xyz is
                   already per-building mean-centred at the .h5 level, so no
                   extra centring needed here -- matches STPLS3D's own
                   block-mean-centring step, just already done upstream)
  colors         : float32 (N, 4) -- rgb (range ~[-1, 1], STPLS3D's prep does
                   rgb/127.5 - 1 for its 0-255 rgb; HFX3D's rgb is already
                   [0,1], so the equivalent transform is rgb*2 - 1) with
                   intensity ([0,1] already) appended as a 4th channel.
 locks the input to 7 channels
                   [x,y,z,r,g,b,intensity] across every benchmark model --
                   ISBNet's own "feat" tensor is rgb-only by default (3ch),
                   same class of "+1 in_channels, wire intensity" edit every
                   other model needed. custom.py's train-time jitter
                   (`feat += torch.randn(3) * 0.1`) is patched to
                   `torch.randn(feat.shape[1])` to match (isbnet/data/custom.py).
  sem_labels     : int64 (N), DENSE STUFF-FIRST remap (RAW_TO_DENSE below),
                   NOT raw HFX3D 0-11 ids -- same table used by
                   mask3d_hfx3d_converter.py and the OneFormer3D converter,
                   reused here for cross-model consistency. Required because
                   ISBNet's get_instance_info() (isbnet/model/model_utils.py)
                   does `instance_cls[i] = semantic_labels[i] - label_shift`
                   for every non-ignore instance -- correct only if stuff
                   classes are a CONTIGUOUS PREFIX of the dense id space.
                   HFX3D's raw stuff ids (wall=0, terrain=6, roof=7, other=9)
                   are not contiguous, hence the remap.
  instance_labels: int64 (N), -100 = ignore (matches ISBNet's own ignore
                   value, confirmed in isbnet/model/model_utils.py:
                   `instance_cls[instance_cls != -100] -= label_shift`).
                   Two things collapse to -100 here, same as STPLS3D's own
                   prep script's `disableInstanceBySemantic_labels` step:
                     (a) every point whose dense semantic id is a stuff
                         class (id < LABEL_SHIFT=4) -- instance-seg is only
                         scored on the 8 thing classes;
                     (b) every point HFX3D's own preprocessing already
                         marked ignore (raw labels/instance == -1 -- this is
                         the protocol's locked 200-pt min-instance-size
                         filter, already baked into the source .h5 files,
                         not redone here).
                   Remaining thing-class instance ids are RENUMBERED to a
                   dense 0..K-1 range (required: ISBNet's get_instance_info
                   does `instance_num = instance_labels.max() + 1` and then
                   iterates every id in range(instance_num) unconditionally
                   -- a gap in the id sequence produces an empty per-instance
                   slice and get_instance_info's `.min(dim=0)` on an empty
                   tensor raises, not just returns garbage. Verified by
                   reading model_utils.py directly, not assumed).

spp (superpoints) are NOT written to these .pth files -- isbnet/data/hfx3d.py
follows stpls3d.py's own precedent exactly (`spp = np.arange(xyz.shape[0])`,
a dummy per-point id) since HFX3D's real superpoint field is already known
to be non-ingestible at this scale and ISBNet's own STPLS3D
config runs with use_spp_pool=False, so spp is loaded but never consumed by
the model in this setup.

HOW TO RUN:
    Requires only h5py, numpy, torch (no ISBNet container needed):

        python isbnet_hfx3d_converter.py \\
            --data_root <HFX3D_Instance> \\
            --out_root  <hfx3d_isbnet> --splits train
"""

import argparse
import glob
import os

import h5py
import numpy as np
import torch

# Stuff classes first -- MUST match mask3d_hfx3d_converter.py /
# oneformer3d_hfx3d_converter.py's DENSE_CLASS_NAMES/RAW_TO_DENSE exactly.
DENSE_CLASS_NAMES = [
    "wall", "terrain", "roof", "other",                                # stuff (4)
    "window", "door", "balcony", "vegetation", "stairs", "blinds",
    "column", "arch",                                                  # thing (8)
]
RAW_TO_DENSE = {
    0: 0, 6: 1, 7: 2, 9: 3,                    # wall, terrain, roof, other
    1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 8: 9, 10: 10, 11: 11,
}
LABEL_SHIFT = 4  # number of stuff classes -- matches ISBNet's semantic_classes - instance_classes
CHUNK = 4_000_000  # points per read/write chunk


def convert_split(data_root, out_root, split):
    in_dir = os.path.join(data_root, split)
    out_dir = os.path.join(out_root, split)
    os.makedirs(out_dir, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(in_dir, "*.h5")))
    class_point_counts = np.zeros(len(DENSE_CLASS_NAMES), dtype=np.int64)

    for h5_path in h5_files:
        name = os.path.splitext(os.path.basename(h5_path))[0]
        base = os.path.join(out_dir, name)
        done = os.path.join(out_dir, name + ".done")
        if os.path.isfile(done):
            # Resumable (a login-node process limit killed the first run mid-split):
            # a building is only marked done after all four arrays are written.
            sem_done = np.load(base + "_sem.npy", mmap_mode="r")
            class_point_counts += np.bincount(sem_done, minlength=len(DENSE_CLASS_NAMES))
            print(f"[{split}] {name}: already converted, counted", flush=True)
            continue

        # Chunked (a whole 45M-point building in RAM can get OOM-killed).
        # Two passes over the h5: (1) collect the kept
        # thing-instance ids, (2) remap and write through np.lib.format
        # memmaps. Output is identical to the original whole-array version.
        sem_lut = np.zeros(max(RAW_TO_DENSE.keys()) + 1, dtype=np.int64)
        for raw_id, dense_id in RAW_TO_DENSE.items():
            sem_lut[raw_id] = dense_id
        with h5py.File(h5_path, "r") as f:
            n = f["point_cloud/xyz"].shape[0]
            chunks = [(s, min(s + CHUNK, n)) for s in range(0, n, CHUNK)]

            # Ignore: HFX3D's own -1 (the 200-pt min-instance-size filter
            # already applied in the released file) OR any
            # stuff-class point (dense id < LABEL_SHIFT) -> -100.
            remaining_ids = np.zeros(0, dtype=np.int64)
            for s, e in chunks:
                sem = sem_lut[f["labels/semantic"][s:e].astype(np.int64)]
                inst = f["labels/instance"][s:e].astype(np.int64)
                keep = (inst != -1) & (sem >= LABEL_SHIFT)
                remaining_ids = np.union1d(remaining_ids, np.unique(inst[keep]))
            # Dense 0..K-1 renumbering (ISBNet's get_instance_info needs no gaps).
            inst_lut = np.full(int(remaining_ids.max()) + 1 if len(remaining_ids) else 1, -100, dtype=np.int64)
            inst_lut[remaining_ids] = np.arange(len(remaining_ids), dtype=np.int64)

            om = np.lib.format.open_memmap
            out_xyz = om(base + "_xyz.npy", mode="w+", dtype=np.float32, shape=(n, 3))
            out_feat = om(base + "_feat.npy", mode="w+", dtype=np.float32, shape=(n, 4))
            out_sem = om(base + "_sem.npy", mode="w+", dtype=np.int64, shape=(n))
            out_inst = om(base + "_inst.npy", mode="w+", dtype=np.int64, shape=(n))
            n_ign = 0
            for s, e in chunks:
                out_xyz[s:e] = f["point_cloud/xyz"][s:e].astype(np.float32)
                rgb = f["point_cloud/rgb"][s:e].astype(np.float32)
                intensity = f["point_cloud/intensity"][s:e].astype(np.float32)
                # released scales, no reinterpretation.
                colors = np.concatenate([rgb, intensity.reshape(-1, 1)], axis=1)
                assert np.all(np.isfinite(colors)) and colors.min() >= 0.0 and colors.max() <= 1.0, name
                out_feat[s:e] = colors
                sem = sem_lut[f["labels/semantic"][s:e].astype(np.int64)]
                inst = f["labels/instance"][s:e].astype(np.int64)
                class_point_counts += np.bincount(sem, minlength=len(DENSE_CLASS_NAMES))
                keep = (inst != -1) & (sem >= LABEL_SHIFT)
                dense = np.full(e - s, -100, dtype=np.int64)
                dense[keep] = inst_lut[inst[keep]]
                out_sem[s:e] = sem
                out_inst[s:e] = dense
                n_ign += int(np.count_nonzero((inst == -1) & (sem >= LABEL_SHIFT)))
            for a_ in (out_xyz, out_feat, out_sem, out_inst):
                a_.flush()
            del out_xyz, out_feat, out_sem, out_inst
        open(done, "w").close()
        print(
            f"[{split}] {name}: {n:,} pts, {len(remaining_ids)} thing instances, "
            f"{n_ign:,} thing pts with instance -1 (-> -100 ignore) -> {base}_*.npy",
            flush=True,
        )

    return class_point_counts


def write_semantic_weight(out_root, class_point_counts):
    """Median-frequency-style inverse weighting from REAL train-split point
    counts, same principle as SoftGroup's own class_numpoint_mean/
    semantic_weight (protocol: recalibrate borrowed defaults using real data,
    never inherit another dataset's numbers as-is)."""
    counts = class_point_counts.astype(np.float64)
    counts[counts == 0] = 1.0  # avoid div-by-zero for any unseen class
    median = np.median(counts)
    weight = np.clip(median / counts, 0.1, 50.0)
    path = os.path.join(out_root, "semantic_weight.txt")
    with open(path, "w") as f:
        f.write("# class_name: point_count -> weight (median-frequency, train split only)\n")
        for name, cnt, w in zip(DENSE_CLASS_NAMES, class_point_counts, weight):
            f.write(f"{name}: {cnt} -> {w:.4f}\n")
        f.write("\nsemantic_weight list (paste into config):\n")
        f.write("[" + ", ".join(f"{w:.4f}" for w in weight) + "]\n")
    print(f"-> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="HFX3D_Instance root (train/val/test subdirs of .h5)")
    parser.add_argument("--out_root", required=True, help="ISBNet dataset/hfx3d output root")
    parser.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val", "test"])
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    for split in args.splits:
        counts = convert_split(args.data_root, args.out_root, split)
        if split == "train":
            write_semantic_weight(args.out_root, counts)


if __name__ == "__main__":
    main()
