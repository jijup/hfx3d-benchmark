"""
Review-2 metrics: compare original vs reviewed classification labels.

Reads each building's frozen baseline (labels/semantic_original, written by
import_instances.py --classification on the first class edit) against the
current labels (labels/semantic), and reports the same metrics the old
utils/review_las_files.py produced — now sourced from the HDF5:

  - overall accuracy   (fraction of points unchanged)
  - Cohen's Kappa      (agreement between original and reviewed)
  - per-class IoU      (CHANGED classes only, matching the old script)
  - confusion matrix   (original -> reviewed; which class became which)
  - class distributions before / after, and #points changed

Reports per building and a dataset-level aggregate, and writes a JSON summary.

Buildings with no labels/semantic_original (never class-reviewed) are skipped
with a note — there is nothing to compare yet.

Usage:
    python scripts/review_metrics.py [--root Output] [--only BLD003]
                                     [--out Output/review_metrics.json]
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
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
except ImportError:
    sys.exit("scikit-learn not found. Install with: pip install scikit-learn")

SPLITS = ["train", "val", "test"]
NUM_CLASSES = 12
CLASS_NAMES = [
    "wall", "window", "door", "balcony", "vegetation", "stairs",
    "terrain", "roof", "blinds", "other", "column", "arch",
]


def changed_class_iou(orig, mod):
    """Per-class IoU for classes whose point count changed (old-script rule).
    IoU(c) = |orig==c & mod==c| / |orig==c | mod==c|."""
    out = {}
    changed = []
    for c in range(NUM_CLASSES):
        n_orig = int((orig == c).sum())
        n_mod = int((mod == c).sum())
        if n_orig == n_mod:
            continue                       # distribution unchanged -> skip
        changed.append(c)
        inter = int(((orig == c) & (mod == c)).sum())
        union = int(((orig == c) | (mod == c)).sum())
        out[c] = round(inter / union, 4) if union > 0 else 0.0
    return out, changed


def metrics(orig, mod):
    acc = float(accuracy_score(orig, mod))
    try:
        kappa = float(cohen_kappa_score(orig, mod))
    except Exception:
        kappa = float("nan")
    iou, changed = changed_class_iou(orig, mod)
    mean_iou = float(np.mean(list(iou.values()))) if iou else 0.0
    dist_orig = {int(c): int((orig == c).sum()) for c in np.unique(orig)}
    dist_mod = {int(c): int((mod == c).sum()) for c in np.unique(mod)}
    cm = confusion_matrix(orig, mod, labels=list(range(NUM_CLASSES)))
    # off-diagonal reassignments: original class -> reviewed class
    reassign = {f"{i}->{j}": int(cm[i, j])
                for i in range(NUM_CLASSES) for j in range(NUM_CLASSES)
                if i != j and cm[i, j] > 0}
    return {
        "points": int(len(orig)),
        "points_changed": int((orig != mod).sum()),
        "overall_accuracy": round(acc, 6),
        "cohen_kappa": round(kappa, 6),
        "mean_iou_changed": round(mean_iou, 6),
        "iou_changed_classes": {int(k): v for k, v in iou.items()},
        "changed_classes": [int(c) for c in changed],
        "reassignments": reassign,
        "confusion_matrix": cm.tolist(),
        "dist_original": dist_orig,
        "dist_reviewed": dist_mod,
    }


def print_building(name, m):
    print(f"\n=== {name} ===")
    print(f"  points: {m['points']:,}  changed: {m['points_changed']:,} "
          f"({m['points_changed']/m['points']:.4%})")
    print(f"  overall accuracy : {m['overall_accuracy']:.4%}")
    print(f"  Cohen's Kappa    : {m['cohen_kappa']:.4f}")
    print(f"  mean IoU (changed): {m['mean_iou_changed']:.4f}")
    if m["iou_changed_classes"]:
        print("  IoU for changed classes:")
        for c, v in sorted(m["iou_changed_classes"].items()):
            print(f"    {c:2d} {CLASS_NAMES[c]:<11}: {v:.4f}")
    else:
        print("  (no class distribution changes)")
    if m["reassignments"]:
        top = sorted(m["reassignments"].items(), key=lambda kv: -kv[1])[:5]
        print("  top reassignments (original -> reviewed):")
        for k, v in top:
            i, j = (int(x) for x in k.split("->"))
            print(f"    {CLASS_NAMES[i]} -> {CLASS_NAMES[j]}: {v:,}")


def main():
    ap = argparse.ArgumentParser(description="Original vs reviewed classification metrics")
    ap.add_argument("--root", default="Output", help="HDF5 root (default: Output)")
    ap.add_argument("--only", default=None, help="Only files containing this string")
    ap.add_argument("--out", default=None,
                    help="JSON summary path (default: <root>/review_metrics.json)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out_path = args.out or os.path.join(root, "review_metrics.json")

    per_building = []
    all_orig, all_mod = [], []          # for the dataset aggregate
    skipped = []

    for split in SPLITS:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for fn in sorted(f for f in os.listdir(split_dir) if f.endswith(".h5")):
            if args.only and args.only not in fn:
                continue
            path = os.path.join(split_dir, fn)
            name = os.path.splitext(fn)[0]
            with h5py.File(path, "r") as h:
                if "labels/semantic_original" not in h:
                    skipped.append(name)
                    continue
                orig = h["labels/semantic_original"][:].astype(np.int64)
                mod = h["labels/semantic"][:].astype(np.int64)
            m = metrics(orig, mod)
            m["name"], m["split"] = name, split
            per_building.append(m)
            all_orig.append(orig)
            all_mod.append(mod)
            print_building(name, m)

    aggregate = None
    if all_orig:
        o = np.concatenate(all_orig)
        d = np.concatenate(all_mod)
        aggregate = metrics(o, d)
        aggregate["buildings"] = len(per_building)
        print("\n" + "=" * 50)
        print(f"DATASET AGGREGATE  ({len(per_building)} reviewed building(s))")
        print("=" * 50)
        print_building("ALL", aggregate)

    if skipped:
        print(f"\nSkipped (no baseline yet): {', '.join(skipped)}")
    if not per_building:
        print("\nNo reviewed buildings found. Run import_instances.py "
              "--classification first to create baselines.")

    with open(out_path, "w") as f:
        json.dump({"per_building": per_building, "aggregate": aggregate,
                   "skipped": skipped}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
