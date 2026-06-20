"""
Import hand-edited instance labels from a CloudCompare PLY back into the HDF5.

Companion to generate_instances.py. That script exports an inspection PLY with
real RGB plus three scalar fields: scalar_Classification, scalar_Instance, and
scalar_PointIndex. You relabel scalar_Instance in CloudCompare (merge = same
value, split = different value), save as PLY, then run this to write the result
into labels/instance of the matching HDF5 file.

Mapping is by scalar_PointIndex (the original 0..N-1 HDF5 row), so it is robust
to any reordering / splitting / partial deletion CloudCompare does. Points not
present in the PLY keep instance 0. The ids you assign in CloudCompare are
PRESERVED as-is (merging into 35 stays 35; gaps are fine) — pass --repack only
as a final step to compact them to 1..K. Optionally re-imports edited semantics.

Usage:
    # single file -> its matching HDF5 (resolved from --root by stem)
    python scripts/import_instances.py edited.ply --root Output
    # explicit target
    python scripts/import_instances.py edited.ply --h5 Output/train/HFX_BLD003_ZEB_CLEAN.h5
    # also import edited scalar_Semantic
    python scripts/import_instances.py edited.ply --root Output --semantic
"""

import os
import sys
import argparse
import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py not found. Install with: pip install h5py")

try:
    from sklearn.metrics import accuracy_score, cohen_kappa_score
except ImportError:
    accuracy_score = cohen_kappa_score = None

# Must match generate_instances.py IDX_BASE.
IDX_BASE = 100_000

PLY_TYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
    "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_ply(path):
    """Return a dict {property_name: 1D ndarray}. Handles ascii and
    binary_little_endian PLY with a single 'vertex' element."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            sys.exit(f"{path}: not a PLY file")
        fmt = None
        n = None
        props = []  # (name, numpy_dtype)
        while True:
            line = f.readline()
            if not line:
                sys.exit(f"{path}: header ended unexpectedly")
            parts = line.split()
            if not parts:
                continue
            kw = parts[0]
            if kw == b"format":
                fmt = parts[1].decode()
            elif kw == b"element":
                if parts[1] == b"vertex":
                    n = int(parts[2])
                else:
                    n = n  # ignore other elements' counts for property capture
            elif kw == b"property":
                if parts[1] == b"list":
                    sys.exit(f"{path}: list properties not supported")
                t = PLY_TYPES.get(parts[1].decode())
                if t is None:
                    sys.exit(f"{path}: unknown property type {parts[1]!r}")
                props.append((parts[2].decode(), t))
            elif kw == b"end_header":
                break

        names = [p[0] for p in props]
        if fmt == "ascii":
            try:                                  # pandas is ~20x faster on big files
                import pandas as pd
                data = pd.read_csv(f, sep=r"\s+", header=None, nrows=n,
                                   dtype=np.float64).to_numpy()
            except ImportError:
                data = np.loadtxt(f, dtype=np.float64, max_rows=n)
            if data.ndim == 1:
                data = data[None, :]
            return {nm: data[:, i] for i, nm in enumerate(names)}
        elif fmt == "binary_little_endian":
            dt = np.dtype([(p[0], "<" + p[1]) for p in props])
            rec = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
            return {nm: rec[nm] for nm in names}
        else:
            sys.exit(f"{path}: unsupported format {fmt!r} (save as ASCII PLY)")


def resolve_h5(ply_path, root):
    stem = os.path.basename(ply_path)
    for suffix in ("_instances_vis", "_instances", "_edited"):
        if stem.endswith(suffix + ".ply"):
            stem = stem[: -len(suffix + ".ply")]
            break
    else:
        stem = os.path.splitext(stem)[0]
    for split in ("train", "val", "test"):
        cand = os.path.join(root, split, stem + ".h5")
        if os.path.isfile(cand):
            return cand
    sys.exit(f"Could not find a matching .h5 for '{stem}' under {root}. "
             f"Pass --h5 explicitly.")


def build_index(cols, h5_path, n):
    """Recover the per-point HDF5 row for every PLY point. Returns (idx, method).

    Preference order:
      1. scalar_PointIdxHi/Lo  -> hi*IDX_BASE + lo  (exact under float32)
      2. scalar_PointIndex     -> used only if not corrupted by float32
      3. XYZ nearest-neighbour -> fallback that recovers older/corrupted files
    """
    if "scalar_PointIdxHi" in cols and "scalar_PointIdxLo" in cols:
        hi = np.rint(cols["scalar_PointIdxHi"]).astype(np.int64)
        lo = np.rint(cols["scalar_PointIdxLo"]).astype(np.int64)
        return hi * IDX_BASE + lo, "hi/lo"

    if "scalar_PointIndex" in cols:
        idx = np.rint(cols["scalar_PointIndex"]).astype(np.int64)
        ok = (idx.min() >= 0 and idx.max() < n
              and len(np.unique(idx)) == len(idx))
        if ok:
            return idx, "index"
        print("  WARNING: scalar_PointIndex is corrupted (float32 precision on "
              ">16.7M points) — falling back to XYZ matching.")

    if not all(k in cols for k in ("x", "y", "z")):
        sys.exit("No usable index and no x/y/z in PLY — cannot map points.")
    from scipy.spatial import cKDTree
    with h5py.File(h5_path, "r") as h:
        hx = h["point_cloud/xyz"][:]
    ply_xyz = np.column_stack([cols["x"], cols["y"], cols["z"]]).astype(np.float64)
    _, idx = cKDTree(hx).query(ply_xyz, k=1)
    return idx.astype(np.int64), "xyz"


def replace_dataset(grp, name, data, dtype):
    """Crash-safe replace: write a temp dataset, then swap, so the old data is
    never deleted until the new data is fully written."""
    tmp = name + "__tmp"
    if tmp in grp:
        del grp[tmp]
    grp.create_dataset(tmp, data=data, dtype=dtype, compression="gzip")
    if name in grp:
        del grp[name]
    grp.move(tmp, name)


def repack(ids):
    """Map arbitrary positive ids to contiguous 1..K; keep 0 as 0."""
    out = np.zeros_like(ids, dtype=np.int32)
    uniq = np.unique(ids[ids > 0])
    for new, old in enumerate(uniq, start=1):
        out[ids == old] = new
    return out, len(uniq)


def main():
    ap = argparse.ArgumentParser(description="Import edited instance labels into HDF5")
    ap.add_argument("ply", help="Edited PLY exported from CloudCompare")
    ap.add_argument("--root", default="Output", help="HDF5 root (default: Output)")
    ap.add_argument("--h5", default=None, help="Target .h5 (overrides --root lookup)")
    ap.add_argument("--classification", "--semantic", dest="classification",
                    action="store_true",
                    help="Also import edited scalar_Classification back into "
                         "labels/semantic (for review 2 / class-label fixes)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing")
    ap.add_argument("--repack", action="store_true",
                    help="Compact instance ids to a gap-free 1..K. OFF by "
                         "default so the ids you assign in CC stay stable "
                         "(e.g. merging into 35 keeps 35). Use this only as a "
                         "final step when freezing the dataset.")
    args = ap.parse_args()

    cols = read_ply(args.ply)
    if "scalar_Instance" not in cols:
        sys.exit(f"{args.ply}: missing 'scalar_Instance'. Re-export with "
                 f"generate_instances.py and keep its scalar fields.")

    inst_in = np.rint(cols["scalar_Instance"]).astype(np.int64)
    inst_in[inst_in < 0] = 0

    h5_path = args.h5 or resolve_h5(args.ply, args.root)
    with h5py.File(h5_path, "r") as h:
        n = int(h["labels/semantic"].shape[0])

    idx, method = build_index(cols, h5_path, n)
    if idx.min() < 0 or idx.max() >= n:
        sys.exit(f"Point index out of range [0,{n}); wrong file pairing?")
    print(f"  point mapping: {method}")

    # Scatter edited values onto a full-length array; unseen points stay 0.
    full = np.zeros(n, dtype=np.int64)
    full[idx] = inst_in
    if args.repack:
        instance, k = repack(full)          # gap-free 1..K (final freeze only)
    else:
        instance = full.astype(np.int32)    # preserve the ids assigned in CC
        k = int(len(np.unique(full[full > 0])))
    missing = n - len(np.unique(idx))

    print(f"PLY points: {len(idx):,}  | HDF5 points: {n:,}  | "
          f"not in PLY (-> instance 0): {missing:,}")
    print(f"Instances: {k}  | max id: {int(instance.max())}"
          f"{'  (re-packed 1..K)' if args.repack else '  (ids preserved)'}")

    sem_field = cols.get("scalar_Classification", cols.get("scalar_Semantic"))
    sem_before = None
    has_orig = True
    if args.classification:
        if sem_field is None:
            sys.exit("--classification given but PLY has no scalar_Classification")
        with h5py.File(h5_path, "r") as h:
            sem_before = h["labels/semantic"][:].astype(np.int64)
            has_orig = "labels/semantic_original" in h
        sem_full = sem_before.copy()
        sem_full[idx] = np.rint(sem_field).astype(np.int64)
        n_changed = int((sem_before != sem_full).sum())
        msg = f"Classification: {n_changed:,} point label(s) changed"
        if accuracy_score is not None:
            acc = accuracy_score(sem_before, sem_full)
            try:
                kap = cohen_kappa_score(sem_before, sem_full)
            except Exception:
                kap = float("nan")
            msg += f" | this-import accuracy {acc:.4%} | kappa {kap:.4f}"
        print(msg)
        if not has_orig:
            print("  (freezing labels/semantic_original baseline for review metrics)")
    else:
        sem_full = None

    if args.dry_run:
        print("dry-run: nothing written.")
        return

    with h5py.File(h5_path, "a") as h:
        grp = h["labels"]
        if sem_full is not None:
            # Freeze the pre-review labels once, before overwriting semantic, so
            # review_metrics.py can always compare original -> reviewed.
            if not has_orig and "semantic_original" not in grp:
                grp.create_dataset("semantic_original",
                                   data=sem_before.astype(np.uint8),
                                   dtype="uint8", compression="gzip")
            replace_dataset(grp, "semantic", sem_full.astype(np.uint8), "uint8")
        replace_dataset(grp, "instance", instance, "int32")
        grp.attrs["num_instances"] = k
        grp.attrs["instance_source"] = os.path.basename(args.ply)
    print(f"Wrote labels/instance ({k} instances) -> {h5_path}")


if __name__ == "__main__":
    main()
