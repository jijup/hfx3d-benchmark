"""
Generate per-building instance IDs for the HFX3D benchmark.

Reads the normalized HDF5 files produced by preprocessing/las_to_hdf5.py and
adds an `labels/instance` dataset to each, in place. Point order is identical
to the source LAS (las_to_hdf5 never reorders), so instance IDs align 1:1 with
the existing `labels/semantic` and `point_cloud/*` datasets.

Instance ID convention (scene-local, ScanNet/S3DIS-style):
  - IDs are unique WITHIN a building only (not shared across buildings).
  - A single running counter 1..K spans all classes in the building.
  - 0 = "no instance" (clusters smaller than --min-points / unlabelled).
  - semantic and instance are kept as SEPARATE fields. The pair
    (semantic, instance) identifies an object. We do NOT pack class*1000+inst.

Clustering: per-class connected components at a fixed metric radius (--eps),
computed on a voxel-downsampled copy for speed, then propagated back to full
resolution by nearest voxel. RGB is never modified.

Visualization (--vis): writes <name>_instances_vis.ply next to a vis folder
with a distinct random colour per instance, for eyeballing in CloudCompare.
This file is for viewing only — the canonical data lives in the HDF5.

Usage:
    python scripts/generate_instances.py [--root Output] [--eps 0.2]
        [--voxel 0.05] [--min-points 50] [--no-vis]
"""

import os
import sys
import json
import argparse
import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py not found. Install with: pip install h5py")

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    sys.exit("scikit-learn not found. Install with: pip install scikit-learn")

from scipy.spatial import cKDTree

NUM_CLASSES = 12
SPLITS = ["train", "val", "test"]

# CloudCompare stores scalar fields as float32, which only represents integers
# exactly up to 2**24 (~16.7M). Buildings exceed that, so the per-point index is
# split into Hi/Lo parts (both << 2**24) and reassembled on import.
IDX_BASE = 100_000


def voxel_downsample(pts, voxel):
    """Return (representative_coords, inverse_index) for a voxel grid.

    inverse_index[i] gives the representative index for full-res point i, so
    full_labels = rep_labels[inverse_index] propagates cluster labels back.
    """
    keys = np.floor(pts / voxel).astype(np.int64)
    _, first_idx, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    reps = pts[first_idx]
    return reps, inverse


def cluster_class(pts, eps, voxel, min_points):
    """Connected components at radius `eps` (metric). Returns labels 1..k,
    with 0 for points in clusters smaller than `min_points`."""
    # Downsample for speed; cluster the representatives, then propagate.
    if voxel and voxel > 0 and pts.shape[0] > 20_000:
        reps, inverse = voxel_downsample(pts, voxel)
    else:
        reps, inverse = pts, np.arange(pts.shape[0])

    # min_samples=1 makes DBSCAN equivalent to single-link connected
    # components at radius eps (every point is a core point; no DBSCAN noise).
    rep_labels = DBSCAN(eps=eps, min_samples=1, algorithm="kd_tree").fit_predict(reps)
    labels = rep_labels[inverse]  # 0-based, contiguous from DBSCAN

    # Shift to 1-based and drop clusters below the full-res min_points.
    labels = labels + 1
    counts = np.bincount(labels)
    small = np.where(counts < min_points)[0]
    if small.size:
        labels[np.isin(labels, small)] = 0

    # Re-pack surviving labels to a contiguous 1..k range.
    uniq = np.unique(labels[labels > 0])
    remap = np.zeros(labels.max() + 1, dtype=np.int64)
    remap[uniq] = np.arange(1, len(uniq) + 1)
    return remap[labels]


def build_instances(xyz, semantic, eps, voxel, min_points, octree_level=None):
    """Assign a building-global running instance ID across all classes.
    Returns int32 array; 0 = no instance.

    If octree_level is set, the per-class radius is the old script's adaptive
    value `class_bbox_max / 2**octree_level` instead of the fixed `eps`."""
    instance = np.zeros(xyz.shape[0], dtype=np.int32)
    next_id = 1
    per_class = {}

    for cls in range(NUM_CLASSES):
        mask = semantic == cls
        n = int(mask.sum())
        if n == 0:
            continue
        pts = xyz[mask]
        if octree_level is not None:
            extent = float(np.ptp(pts, axis=0).max())
            cls_eps = extent / (2 ** octree_level) if extent > 0 else eps
        else:
            cls_eps = eps
        # Keep the clustering voxel finer than the radius, else downsampling
        # would merge across the gap the radius is meant to respect.
        cls_voxel = min(voxel, cls_eps / 2.0) if voxel else voxel
        local = cluster_class(pts, cls_eps, cls_voxel, min_points)
        k = int(local.max())
        if k > 0:
            # Offset local 1..k into the building-global namespace.
            shifted = np.where(local > 0, local + (next_id - 1), 0)
            instance[mask] = shifted
            next_id += k
        per_class[cls] = k
    return instance, next_id - 1, per_class


def reassign_orphans(xyz, semantic, instance):
    """Attach each instance-0 point to the nearest point that has a valid
    instance OF THE SAME CLASS. Points whose class has no valid instance are
    left at 0. Mutates and returns `instance`."""
    for cls in np.unique(semantic):
        cls_mask = semantic == cls
        orphan = cls_mask & (instance == 0)
        valid = cls_mask & (instance > 0)
        if not orphan.any() or not valid.any():
            continue
        tree = cKDTree(xyz[valid])
        _, nn = tree.query(xyz[orphan], k=1)
        instance[orphan] = instance[valid][nn]
    return instance


def replace_dataset(grp, name, data, dtype):
    """Crash-safe dataset replace: write to a temp name, then swap in. The old
    data is never deleted until the new data is fully written, so an interrupted
    run can't leave the dataset missing."""
    tmp = name + "__tmp"
    if tmp in grp:
        del grp[tmp]
    grp.create_dataset(tmp, data=data, dtype=dtype, compression="gzip")
    if name in grp:
        del grp[name]
    grp.move(tmp, name)


def class_colors():
    """(12,3) int RGB from the official palette (configs/palette.py)."""
    import importlib.util
    palette_path = os.path.join(os.path.dirname(__file__), "..", "configs", "palette.py")
    try:
        spec = importlib.util.spec_from_file_location("palette", palette_path)
        pal = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pal)
        hexes = pal.HEX
    except Exception as e:
        sys.exit(f"Could not load palette from {palette_path}: {e}")
    return np.array([[int(h.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
                     for h in hexes], dtype=np.int64)


def instance_colors(num):
    """Distinct RGB per instance via golden-ratio hue spacing (uint8).
    Index 0 = grey (no instance)."""
    golden = 0.618033988749895
    colors = np.zeros((num + 1, 3), dtype=np.int64)
    colors[0] = (60, 60, 60)
    hue = 0.0
    for i in range(1, num + 1):
        hue = (hue + golden) % 1.0
        h = hue * 6.0
        k = int(h)
        f = h - k
        q = 1 - f
        opts = [(1, f, 0), (q, 1, 0), (0, 1, f), (0, q, 1), (f, 0, 1), (1, 0, q)]
        r, g, b = opts[k % 6]
        colors[i] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


def write_vis_ply(path, xyz, rgb01, semantic, instance, color="instance"):
    """Write an editable inspection PLY for CloudCompare.

    RGB channel holds either distinct per-instance colours (color="instance",
    the default — so every object is visibly different the moment you open it)
    or the real building colours (color="real", for context). Either way the
    real colours are safe in the HDF5; this PLY is disposable.

    Scalar fields ride alongside:
        scalar_Classification  class 0-11 (read-only context; = HDF5 semantic)
        scalar_Instance        object id (edit THIS in CC to relabel)
        scalar_PointIdxHi/Lo   original HDF5 row split into two parts (base
                               100000) so float32 storage in CloudCompare keeps
                               it exact. The key that maps your edits back, no
                               matter how CC reorders/splits/deletes. DO NOT edit.

    Save edits from CC as an ASCII PLY, keeping these scalar fields.
    """
    n = xyz.shape[0]
    if color == "real":
        rgb = np.clip(np.rint(rgb01 * 255.0), 0, 255).astype(np.int64)
    elif color == "class":
        rgb = class_colors()[np.clip(semantic, 0, 11).astype(np.int64)]
    else:
        rgb = instance_colors(int(instance.max()))[instance]
    idx = np.arange(n, dtype=np.int64)
    # Binary little-endian PLY: ~5x smaller and far faster than ASCII at the
    # 10-40M point scale. CloudCompare reads it natively.
    dt = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("scalar_Classification", "<i4"),
        ("scalar_Instance", "<i4"),
        ("scalar_PointIdxHi", "<i4"),
        ("scalar_PointIdxLo", "<i4"),
    ])
    arr = np.empty(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    arr["scalar_Classification"] = semantic
    arr["scalar_Instance"] = instance
    arr["scalar_PointIdxHi"] = idx // IDX_BASE
    arr["scalar_PointIdxLo"] = idx % IDX_BASE
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property int scalar_Classification\n"
        "property int scalar_Instance\n"
        "property int scalar_PointIdxHi\n"
        "property int scalar_PointIdxLo\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        arr.tofile(f)


def process_file(h5_path, vis_dir, eps, voxel, min_points, write_vis,
                 reassign, vis_only, color="instance", octree_level=None,
                 from_h5=False):
    name = os.path.splitext(os.path.basename(h5_path))[0]
    with h5py.File(h5_path, "r") as h:
        xyz = h["point_cloud/xyz"][:]
        semantic = h["labels/semantic"][:]
        rgb01 = h["point_cloud/rgb"][:] if "point_cloud/rgb" in h \
            else np.zeros_like(xyz)
        stored = h["labels/instance"][:] if "labels/instance" in h else None

    if from_h5:
        # Re-export the editable PLY from labels already in the HDF5 — does NOT
        # re-cluster, so it preserves any hand-edits already imported.
        if stored is None:
            sys.exit(f"{name}: --from-h5 but no labels/instance present. "
                     f"Run a clustering pass (without --from-h5) first.")
        instance, per_class, vis_only = stored, {}, True
        num_inst = int(len(np.unique(instance[instance > 0])))
        orphans = int((instance == 0).sum())
    else:
        instance, num_inst, per_class = build_instances(
            xyz, semantic, eps, voxel, min_points, octree_level
        )
        if reassign:
            instance = reassign_orphans(xyz, semantic, instance)
        orphans = int((instance == 0).sum())

    # Write instance back into the HDF5 (replace if present). RGB untouched.
    # Skipped in --vis-only mode so the dataset is never modified while tuning.
    if not vis_only:
        with h5py.File(h5_path, "a") as h:
            replace_dataset(h["labels"], "instance", instance, "int32")
            h["labels"].attrs["num_instances"] = num_inst
            h["labels"].attrs["instance_eps"] = eps
            h["labels"].attrs["instance_min_points"] = min_points

    if write_vis:
        os.makedirs(vis_dir, exist_ok=True)
        write_vis_ply(
            os.path.join(vis_dir, f"{name}_instances_vis.ply"),
            xyz, rgb01, semantic, instance, color,
        )

    return {"name": name, "num_instances": num_inst,
            "orphan_points": orphans, "per_class": per_class}


def main():
    ap = argparse.ArgumentParser(description="Generate per-building instance IDs")
    ap.add_argument("--root", default="Output",
                    help="HDF5 root with train/ val/ test/ subdirs (default: Output)")
    ap.add_argument("--eps", type=float, default=0.20,
                    help="Connected-components radius in metres (default: 0.20)")
    ap.add_argument("--voxel", type=float, default=0.05,
                    help="Voxel size for downsampled clustering, metres (default: 0.05; 0 disables)")
    ap.add_argument("--octree-level", type=int, default=None,
                    help="Replicate the old script's adaptive per-class radius "
                         "(class_bbox_max / 2**level) instead of fixed --eps. "
                         "Old script used 8.")
    ap.add_argument("--min-points", type=int, default=50,
                    help="Drop clusters smaller than this (default: 50)")
    ap.add_argument("--no-vis", action="store_true", help="Skip the vis PLY export")
    ap.add_argument("--color", choices=["instance", "real", "class"],
                    default="instance",
                    help="Vis PLY RGB channel: 'instance' = distinct colour per "
                         "object (default, best for instance editing); 'class' = "
                         "official 12-class palette (best for classification "
                         "review); 'real' = true colours")
    ap.add_argument("--no-reassign", action="store_true",
                    help="Leave sub-min-points clusters as instance 0 instead of "
                         "attaching them to the nearest same-class instance")
    ap.add_argument("--vis-only", action="store_true",
                    help="Write only the vis PLY; do NOT modify the HDF5 "
                         "(use this while tuning --eps / --min-points)")
    ap.add_argument("--from-h5", action="store_true",
                    help="Export the editable PLY from the labels/instance "
                         "already in the HDF5 instead of re-clustering. Use to "
                         "get a building's PLY for editing, or to re-view it "
                         "after importing hand-edits.")
    ap.add_argument("--only", default=None,
                    help="Process only files whose name contains this string, "
                         "e.g. --only BLD003")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.exit(f"Root not found: {root}")

    summary = []
    for split in SPLITS:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        vis_dir = os.path.join(root, "instances_vis", split)
        files = sorted(f for f in os.listdir(split_dir) if f.endswith(".h5"))
        if args.only:
            files = [f for f in files if args.only in f]
        for i, fn in enumerate(files, 1):
            path = os.path.join(split_dir, fn)
            print(f"[{split} {i}/{len(files)}] {fn} ...", flush=True)
            info = process_file(
                path, vis_dir, args.eps, args.voxel, args.min_points,
                not args.no_vis, not args.no_reassign, args.vis_only,
                args.color, args.octree_level, args.from_h5,
            )
            info["split"] = split
            summary.append(info)
            print(f"    -> {info['num_instances']} instances, "
                  f"{info['orphan_points']} orphan pts left", flush=True)

    out = os.path.join(root, "instance_stats.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Stats: {out}")


if __name__ == "__main__":
    main()
