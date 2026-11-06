"""
Oracle-semantic clustering baselines for HFX3D instance segmentation: DBSCAN, HDBSCAN, MeanShift, KMeans.

Per thing class (ground-truth semantics), per building:
  1. voxelise the class's points at 5 cm (mean 7-D feature per voxel);
  2. standardise with the TRAINING-split statistics from derive_cluster_hparams.py;
  3. cluster the voxels (subsampling with a fixed seed only above a per-method
     cap, with point-count parameters rescaled by the sampled fraction);
  4. give every point its voxel's label (full-resolution output);
  5. score with the shared evaluator (evaluation/insseg_eval_utils.py).
Scores are constant (unranked). Each method has ONE scale parameter, chosen on
the val split from GRIDS below; the test split is then run once with it.

Usage:
    python models/clustering/eval_cluster_hfx3d.py --method dbscan --grid-index 2 \
        --split val --data-root /path/to/HFX3D_Instance --hparams cluster_hparams.json \
        --out-dir results/clustering
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, HDBSCAN, MeanShift, MiniBatchKMeans

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evaluation"))
from insseg_eval_utils import ScanNetInstanceEvaluator  # noqa: E402

from cluster_common import (  # noqa: E402
    CLASS_NAMES, INSTANCE_IGNORE_INDEX, SEED, SEGMENT_IGNORE_INDEX, THING_CLASSES,
    LazyMasks, list_buildings, load_building, voxelize,
)

# One validation-tuned scale parameter per method (fixed grids).
# If the val optimum lies on a grid edge, the grid is extended in
# that direction (val only) until the optimum is interior; extensions are
# appended so earlier grid indices keep their meaning.
GRIDS = {
    "dbscan": [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],     # multiplier on the train k-distance radius
    "hdbscan": [0.1, 0.2, 0.33, 0.5, 1.0],                   # fraction of the train 25th-pct instance size
    "meanshift": [0.01, 0.02, 0.05, 0.1, 0.3, 0.5],          # estimate_bandwidth quantile
    "kmeans": ["instance_size_p25", "instance_size_median", "instance_size_p75"],  # train instance-size statistic for k
}
# Voxel caps per class per building; above them, a seeded subsample is clustered.
MAX_VOXELS = {"dbscan": 1_500_000, "hdbscan": 100_000, "meanshift": 50_000, "kmeans": 1_500_000}
KNN_K = 6
MIN_CLUSTER_POINTS = 50  # cheap pre-filter; the evaluator's 100-point filter is authoritative
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))


def cluster_class(z, method, value, hp, rng):
    """Cluster standardised voxel features z (V, 7). Returns a label per voxel (-1 = noise)."""
    n_vox = len(z)
    cap = MAX_VOXELS[method]
    if n_vox > cap:
        sub = np.sort(rng.choice(n_vox, cap, replace=False))
        frac = cap / n_vox
    else:
        sub = np.arange(n_vox)
        frac = 1.0
    zs = z[sub]
    min_samples = max(2, round(KNN_K * frac))  # keeps the density threshold under subsampling

    if method == "kmeans":
        k = int(max(1, min(len(zs), round(n_vox / hp[value]))))
        km = MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=min(4096, len(zs)), random_state=SEED)
        km.fit(zs)
        return km.predict(z)

    if method == "dbscan":
        sub_labels = DBSCAN(eps=value * hp["dbscan_eps_base"], min_samples=min(min_samples, len(zs) - 1),
                            algorithm="ball_tree", n_jobs=N_JOBS).fit_predict(zs)
    elif method == "hdbscan":
        mcs = int(max(2, min(len(zs), round(value * hp["instance_size_p25"] * frac))))
        sub_labels = HDBSCAN(min_cluster_size=mcs, min_samples=min(min_samples, len(zs))).fit_predict(zs)
    else:  # meanshift
        sub_labels = MeanShift(bandwidth=hp["meanshift_bandwidth"][str(value)], bin_seeding=True,
                               n_jobs=N_JOBS).fit_predict(zs)

    if frac == 1.0:
        return sub_labels
    labels = np.empty(n_vox, dtype=sub_labels.dtype)
    labels[sub] = sub_labels
    excluded = np.setdiff1d(np.arange(n_vox), sub, assume_unique=True)
    _, nn = cKDTree(zs[:, :3]).query(z[excluded][:, :3], k=1, workers=N_JOBS)
    labels[excluded] = sub_labels[nn]
    return labels


def cluster_building(xyz, feat, semantic, method, value, hparams, rng):
    """Returns (list of point-index arrays, classes, scores) for one building."""
    pred_idx, pred_classes = [], []
    for c in THING_CLASSES:
        idx = np.flatnonzero(semantic == c)
        if len(idx) < MIN_CLUSTER_POINTS:
            continue
        if str(c) not in hparams["classes"]:
            print(f"    WARNING: no training hyperparameters for {CLASS_NAMES[c]}; skipped", flush=True)
            continue
        hp = hparams["classes"][str(c)]
        vfeat, p2v = voxelize(xyz[idx], feat[idx])
        z = (vfeat - np.asarray(hp["feat_mean"], dtype=np.float32)) / np.asarray(hp["feat_std"], dtype=np.float32)
        point_labels = cluster_class(z, method, value, hp, rng)[p2v]

        order = np.argsort(point_labels, kind="stable")
        labs, starts = np.unique(point_labels[order], return_index=True)
        ends = np.append(starts[1:], len(order))
        for lab, s, e in zip(labs, starts, ends):
            if lab == -1 or e - s < MIN_CLUSTER_POINTS:
                continue
            pred_idx.append(idx[order[s:e]])
            pred_classes.append(c)
        print(f"    {CLASS_NAMES[c]}: {len(idx)} pts, {len(z)} voxels", flush=True)
    return pred_idx, pred_classes, [1.0] * len(pred_idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(GRIDS), required=True)
    parser.add_argument("--grid-index", type=int, required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--hparams", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--save-pred", action="store_true", help="save per-point instance ids (qualitative figures)")
    args = parser.parse_args()

    value = GRIDS[args.method][args.grid_index]
    with open(args.hparams) as f:
        hparams = json.load(f)
    rng = np.random.RandomState(SEED)
    evaluator = ScanNetInstanceEvaluator(CLASS_NAMES, SEGMENT_IGNORE_INDEX, INSTANCE_IGNORE_INDEX)

    files = list_buildings(args.data_root, args.split)
    assert len(files) == 4, f"expected 4 {args.split} buildings, found {len(files)}"
    tag = f"{args.method}_g{args.grid_index}_{args.split}"
    os.makedirs(args.out_dir, exist_ok=True)
    scenes, n_pred, t0 = [], {}, time.time()
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        print(f"{name} ({args.method}, value={value})", flush=True)
        xyz, feat, semantic, instance = load_building(fp)
        pred_idx, pred_classes, pred_scores = cluster_building(xyz, feat, semantic, args.method, value, hparams, rng)
        n_pred[name] = len(pred_idx)
        if args.save_pred:
            inst = np.full(len(semantic), -1, dtype=np.int32)
            for i, p in enumerate(pred_idx):
                inst[p] = i
            np.save(os.path.join(args.out_dir, f"{tag}_{name}_instance.npy"), inst)
            np.save(os.path.join(args.out_dir, f"{tag}_{name}_classes.npy"), np.asarray(pred_classes, dtype=np.int32))
        gt, pred = evaluator.associate_instances(
            LazyMasks(pred_idx, len(semantic)), pred_classes, pred_scores, semantic, instance
        )
        scenes.append(dict(gt=gt, pred=pred))

    ap = evaluator.evaluate_matches(scenes)
    result = dict(
        protocol_version="1.1", method=args.method, split=args.split, grid_index=args.grid_index,
        value=value, mAP=ap["all_ap"], AP50=ap["all_ap_50"], AP25=ap["all_ap_25"],
        per_class={n: ap["classes"][n] for n in evaluator.valid_class_names},
        n_pred_instances=n_pred, runtime_s=round(time.time() - t0, 1),
    )
    with open(os.path.join(args.out_dir, f"{tag}.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"=== {args.method} value={value} {args.split}: mAP/AP50/AP25 "
          f"{result['mAP']:.4f}/{result['AP50']:.4f}/{result['AP25']:.4f}")
    for n in evaluator.valid_class_names:
        c = ap["classes"][n]
        print(f"  {n:12s} AP/AP50/AP25: {c['ap']:.4f}/{c['ap50']:.4f}/{c['ap25']:.4f}")


if __name__ == "__main__":
    main()
