"""
Normalize the released HFX3D HDF5 files into the input form the benchmark
loaders expect.

The dataset on Hugging Face stores RAW values (xyz in scanner metres, rgb as
uint8 0-255, intensity as uint16). The benchmark's model loaders, however,
expect the normalized input protocol:

    xyz       -> mean-centred per building, metric scale preserved  (float32)
    rgb       -> scaled to [0, 1]                                   (float32)
    intensity -> clipped to the training-split [p1, p99] then scaled to [0, 1]

This script applies exactly that. It is two-pass: p1/p99 are computed once from
the TRAINING split only and applied to every split, so the statistics are
identical across train/val/test and are not contaminated by val/test. All other
groups (labels/semantic, labels/instance, functional/*, metadata/*) are copied
through unchanged, so the output stays usable for every task.

Usage:
    python preprocessing/normalize_hdf5.py <raw_dir> <output_dir>

<raw_dir> is the Hugging Face download with train/ val/ test/ subdirs of *.h5.
Point HFX3D_DATA_ROOT at <output_dir> afterwards. The computed p1/p99 are also
written to <output_dir>/normalization_stats.json for reference.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py is required: pip install h5py")

SPLITS = ("train", "val", "test")


def compute_intensity_percentiles(train_files, sample_per_file=100_000):
    """Global p1/p99 from a random sample of each training file's intensity."""
    samples = []
    rng = np.random.default_rng(42)
    for f in train_files:
        with h5py.File(f, "r") as h:
            if "point_cloud/intensity" not in h:
                continue
            raw = h["point_cloud/intensity"][:].astype(np.float64)
        if raw.size > sample_per_file:
            raw = raw[rng.choice(raw.size, sample_per_file, replace=False)]
        samples.append(raw)
    if not samples:
        print("  warning: no intensity data found; using identity range [0, 1]")
        return 0.0, 1.0
    alli = np.concatenate(samples)
    return float(np.percentile(alli, 1)), float(np.percentile(alli, 99))


def normalize_file(src, dst, p1, p99):
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        # --- point_cloud: the only group we transform ---
        xyz = fin["point_cloud/xyz"][:].astype(np.float64)
        centroid = xyz.mean(axis=0)
        xyz = (xyz - centroid).astype(np.float32)

        pc = fout.create_group("point_cloud")
        pc.create_dataset("xyz", data=xyz, compression="gzip")

        if "point_cloud/rgb" in fin:
            rgb = fin["point_cloud/rgb"][:].astype(np.float32) / 255.0  # uint8 0-255 -> [0,1]
            pc.create_dataset("rgb", data=rgb, compression="gzip")

        if "point_cloud/intensity" in fin:
            inten = fin["point_cloud/intensity"][:].astype(np.float32)
            span = p99 - p1
            inten = (np.clip(inten, p1, p99) - p1) / span if span > 0 else np.zeros_like(inten)
            pc.create_dataset("intensity", data=inten.astype(np.float32), compression="gzip")

        # --- everything else: copy through unchanged ---
        for key in fin:
            if key == "point_cloud":
                continue
            fin.copy(key, fout)

        meta = fout.require_group("metadata")
        if "applied_centroid" in meta:
            del meta["applied_centroid"]
        meta.create_dataset("applied_centroid", data=centroid.astype(np.float64))
        meta.attrs["intensity_p1"] = p1
        meta.attrs["intensity_p99"] = p99
        meta.attrs["normalized"] = True


def main():
    ap = argparse.ArgumentParser(description="Normalize raw HFX3D HDF5 into the benchmark input form.")
    ap.add_argument("raw_dir", help="HF download root with train/ val/ test/ subdirs of *.h5")
    ap.add_argument("output_dir", help="output root (train/ val/ test/ are created)")
    args = ap.parse_args()

    raw, out = Path(args.raw_dir), Path(args.output_dir)
    if not raw.is_dir():
        sys.exit(f"'{raw}' does not exist")

    train_files = sorted((raw / "train").glob("*.h5"))
    if not train_files:
        sys.exit(f"no .h5 files in {raw / 'train'}")
    print(f"Computing intensity p1/p99 from {len(train_files)} training file(s)...")
    p1, p99 = compute_intensity_percentiles(train_files)
    print(f"  p1 = {p1:.4f}, p99 = {p99:.4f}")

    for split in SPLITS:
        src_dir = raw / split
        if not src_dir.exists():
            print(f"skipping {split} (not found)")
            continue
        (out / split).mkdir(parents=True, exist_ok=True)
        files = sorted(src_dir.glob("*.h5"))
        print(f"\nNormalizing {len(files)} {split} file(s)...")
        for src in files:
            dst = out / split / src.name
            normalize_file(str(src), str(dst), p1, p99)
            print(f"  {src.name} -> {dst}")

    with open(out / "normalization_stats.json", "w") as f:
        json.dump({"intensity_p1": p1, "intensity_p99": p99}, f, indent=2)
    print(f"\nDone. Stats written to {out / 'normalization_stats.json'}.")


if __name__ == "__main__":
    main()
