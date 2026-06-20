"""
Repack instance ids to a gap-free 1..K, directly in the HDF5.

During annotation, instance ids are preserved as you assign them (merges/splits
leave gaps, and new objects may use big numbers like 9001). That's fine for
training. When you're DONE with a building, run this to compact its ids to a
clean contiguous 1..K (0 stays 0 = no instance). Grouping is unchanged — only
the id numbers are renumbered.

Operates per building (instance ids are scene-local). No PLY needed.

Note: after repacking a building, any previously exported PLY for it is stale
(its instance numbers won't match). Only repack buildings you've finished, then
re-export with generate_instances.py --from-h5 if you still want to view them.

This ALSO deletes labels/semantic_original (the review-2 baseline) as part of
the final freeze. Run review_metrics.py first to capture the report — once the
baseline is gone the metrics can't be recomputed. Use --dry-run first to see
exactly what will change.

Usage:
    python scripts/repack_instances.py [--root Output] [--only BLD003] [--dry-run]
"""

import os
import sys
import argparse
import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py not found. Install with: pip install h5py")

SPLITS = ["train", "val", "test"]


def replace_dataset(grp, name, data, dtype):
    """Crash-safe replace: write a temp dataset, then swap in."""
    tmp = name + "__tmp"
    if tmp in grp:
        del grp[tmp]
    grp.create_dataset(tmp, data=data, dtype=dtype, compression="gzip")
    if name in grp:
        del grp[name]
    grp.move(tmp, name)


def repack(ids):
    """Map positive ids to a contiguous 1..K (sorted); keep 0 as 0."""
    out = np.zeros_like(ids, dtype=np.int32)
    uniq = np.unique(ids[ids > 0])
    remap = {old: new for new, old in enumerate(uniq, start=1)}
    for old, new in remap.items():
        out[ids == old] = new
    return out, len(uniq)


def main():
    ap = argparse.ArgumentParser(description="Repack instance ids to 1..K in HDF5")
    ap.add_argument("--root", default="Output", help="dataset root (default: Output)")
    ap.add_argument("--only", default=None, help="only files containing this string")
    ap.add_argument("--dry-run", action="store_true",
                    help="preview what would change without writing — run this "
                         "first")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    touched = 0
    for split in SPLITS:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for fn in sorted(f for f in os.listdir(split_dir) if f.endswith(".h5")):
            if args.only and args.only not in fn:
                continue
            path = os.path.join(split_dir, fn)
            with h5py.File(path, "r") as h:
                if "labels/instance" not in h:
                    continue
                inst = h["labels/instance"][:]
                has_orig = "labels/semantic_original" in h

            new, k = repack(inst)
            max_before = int(inst.max())
            already = np.array_equal(new, inst)
            status = "unchanged" if already else f"max {max_before} -> {k}"
            drop = has_orig
            extra = "  + drop baseline" if drop else ""
            print(f"{fn:40s} {k} instances  ({status}){extra}")

            if args.dry_run or (already and not drop):
                continue
            with h5py.File(path, "a") as h:
                if not already:
                    replace_dataset(h["labels"], "instance", new, "int32")
                    h["labels"].attrs["num_instances"] = k
                if drop:
                    del h["labels/semantic_original"]
            touched += 1

    if args.dry_run:
        print("\ndry-run: nothing written.")
    else:
        print(f"\nUpdated {touched} building(s).")


if __name__ == "__main__":
    main()
