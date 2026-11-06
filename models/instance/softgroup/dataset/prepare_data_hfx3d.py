"""
HFX3D -> SoftGroup .pth preprocessing.

SoftGroup's CustomDataset.transform_train() runs dataAugment() then elastic
distortion on the FULL point cloud BEFORE cropping to max_npoint -- fine for
ScanNet/S3DIS room-scale scans, but HFX3D buildings run up to ~47M raw
points, so that ordering would make elastic
distortion (an O(n) gaussian-filtered grid warp) the bottleneck exactly like
it was for Pointcept's PointGroup config before ColumnCrop was moved to run
first there. This script avoids that the same way: pre-tile OFFLINE into
~15m-footprint blocks (matching the tile size, for cross-model
comparability) before any of SoftGroup's own per-block augmentation ever
runs -- each .pth file the dataset loader sees is already crop-sized.

Class-id bookkeeping mirrors Mask3D's converter exactly (same underlying
need: SoftGroup's model internals assume stuff/ignore classes are a
CONTIGUOUS PREFIX of the semantic id space -- see softgroup.py's
`cls_offset = semantic_classes - instance_classes - 1` -- HFX3D's raw order
has stuff at ids 0,6,7,9, not a prefix, so it needs the same dense remap
Mask3D already required, not a new problem):
  RAW_TO_DENSE = {0:0, 6:1, 7:2, 9:3, 1:4, 2:5, 3:6, 4:7, 5:8, 8:9, 10:10, 11:11}
  dense 0-3 = stuff (wall, terrain, roof, other) -- excluded from instance
  dense 4-11 = thing (window, door, balcony, vegetation, stairs, blinds,
               column, arch) -- semantic_classes=12, instance_classes=8,
               label_shift=4, matching Mask3D's label_offset=4 exactly.

Instance labels: HFX3D's raw convention is 1..M with -1 for ignore/no
instance. SoftGroup's CustomDataset (collate_fn) expects 0..M-1 DENSE
per-BLOCK instance ids with -100 for ignore (both "no instance" and "stuff
semantic class" get -100 here, matching STPLS3D's own
disableInstanceBySemantic mechanism).

HOW TO RUN THIS YOURSELF:
    cd /scratch/abdiaziz/hfx3d/repos/SoftGroup
    apptainer exec --nv --cleanenv \
      --bind /scratch/abdiaziz/hfx3d:/scratch/abdiaziz/hfx3d \
      /scratch/abdiaziz/hfx3d/containers/killarney_pointcept.sif \
      python dataset/hfx3d/prepare_data_hfx3d.py \
        --data-root /scratch/abdiaziz/hfx3d/HFX3D_Instance \
        --out-root dataset/hfx3d \
        [--max-buildings N]   # debug: only process the first N buildings/split
"""

import argparse
import os

import h5py
import numpy as np
import torch

RAW_TO_DENSE = {0: 0, 6: 1, 7: 2, 9: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 8: 9, 10: 10, 11: 11}
DENSE_STUFF_IDS = (0, 1, 2, 3)  # dense wall, terrain, roof, other
FOOTPRINT = 15.0  # tile size, matches PointGroup/Mask3D
STRIDE = 15.0  # non-overlapping for train/val blocks (SoftGroup's own
# STPLS3D script also uses non-overlapping grid blocks; full-scene
# overlap+merge is a separate future test-time script, same as the one
# already built for PointGroup/Mask3D -- out of scope for this smoke test)
MIN_BLOCK_POINTS = 10000  # matches STPLS3D script's own threshold


def remap_semantic_dense(raw_semantic):
    dense = np.full_like(raw_semantic, -1)
    for raw_id, dense_id in RAW_TO_DENSE.items():
        dense[raw_semantic == raw_id] = dense_id
    assert (dense >= 0).all(), "unmapped raw semantic id found"
    return dense


def remap_instance_dense_per_block(raw_instance_block, dense_semantic_block):
    """0-indexed consecutive instance ids within this block, -100 for
    ignore (no instance OR stuff-class point), matching SoftGroup's
    CustomDataset/collate_fn convention exactly (see custom.py)."""
    inst = raw_instance_block.copy().astype(np.int64)
    inst[np.isin(dense_semantic_block, DENSE_STUFF_IDS)] = -1  # disableInstanceBySemantic
    unique_ids = np.unique(inst[inst != -1])
    remap = {old: new for new, old in enumerate(unique_ids)}
    out = np.full_like(inst, -100)
    for old, new in remap.items():
        out[inst == old] = new
    return out


def split_blocks(xyz, footprint=FOOTPRINT, stride=STRIDE):
    xy_min = xyz[:, :2].min(0)
    xy_max = xyz[:, :2].max(0)
    xs = np.arange(xy_min[0], xy_max[0] + 1e-6, stride)
    ys = np.arange(xy_min[1], xy_max[1] + 1e-6, stride)
    blocks = []
    for x in xs:
        for y in ys:
            mask = (
                (xyz[:, 0] >= x) & (xyz[:, 0] < x + footprint) &
                (xyz[:, 1] >= y) & (xyz[:, 1] < y + footprint)
            )
            if mask.sum() >= MIN_BLOCK_POINTS:
                blocks.append(mask)
    return blocks


def process_split(h5_files, out_dir, max_buildings=None):
    os.makedirs(out_dir, exist_ok=True)
    if max_buildings:
        h5_files = h5_files[:max_buildings]
    n_blocks_total = 0
    for h5_path in h5_files:
        name = os.path.splitext(os.path.basename(h5_path))[0]
        with h5py.File(h5_path, "r") as f:
            xyz = f["point_cloud/xyz"][:].astype(np.float32)
            rgb = f["point_cloud/rgb"][:].astype(np.float32)
            raw_semantic = f["labels/semantic"][:].reshape(-1).astype(np.int64)
            raw_instance = f["labels/instance"][:].reshape(-1).astype(np.int64)

        dense_semantic = remap_semantic_dense(raw_semantic)
        colors = rgb * 2.0 - 1.0  # HFX3D rgb is [0,1]; SoftGroup expects ~[-1,1]

        blocks = split_blocks(xyz)
        for bi, mask in enumerate(blocks):
            b_xyz = np.ascontiguousarray(xyz[mask] - xyz[mask].mean(0), dtype=np.float32)
            b_colors = np.ascontiguousarray(colors[mask], dtype=np.float32)
            b_sem = dense_semantic[mask].astype(np.int32)
            b_inst = remap_instance_dense_per_block(raw_instance[mask], b_sem)
            out_path = os.path.join(out_dir, f"{name}_{bi}_inst_nostuff.pth")
            torch.save((b_xyz, b_colors, b_sem, b_inst), out_path)
        n_blocks_total += len(blocks)
        print(f"{name}: {xyz.shape[0]} pts -> {len(blocks)} blocks", flush=True)
    print(f"Wrote {n_blocks_total} blocks to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/scratch/abdiaziz/hfx3d/HFX3D_Instance")
    parser.add_argument("--out-root", default="dataset/hfx3d")
    parser.add_argument("--max-buildings", type=int, default=None)
    args = parser.parse_args()

    import glob
    for split, out_name in [("train", "train"), ("val", "val"), ("test", "test")]:
        h5_files = sorted(glob.glob(os.path.join(args.data_root, split, "*.h5")))
        assert h5_files, f"no .h5 files found for split {split} under {args.data_root}"
        process_split(h5_files, os.path.join(args.out_root, out_name), args.max_buildings)


if __name__ == "__main__":
    main()
