"""
Derive every clustering hyperparameter from the TRAINING split only. Run once; the JSON it writes is what
eval_cluster_hfx3d.py reads.

Per thing class, over the 18 training buildings, on 5 cm voxels:
  - feature mean / std (pooled over all training voxels of the class), used to
    standardise the 7-D voxel features identically in every building;
  - DBSCAN radius base: median distance to the 6th nearest neighbour in the
    standardised space (k-distance heuristic; the median over buildings of the
    per-building medians), with min_samples = 6;
  - GT instance size in voxels: 25th percentile and median (HDBSCAN minimum
    cluster size and KMeans k);
  - MeanShift bandwidth for each grid quantile (sklearn estimate_bandwidth,
    median over buildings).

Usage:
    python models/clustering/derive_cluster_hparams.py \
        --data-root /path/to/HFX3D_Instance --out cluster_hparams.json
"""

import argparse
import json

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import estimate_bandwidth

from cluster_common import (
    CLASS_NAMES, SEED, THING_CLASSES, VOXEL_SIZE,
    instance_sizes_in_voxels, list_buildings, load_building, voxelize,
)

KNN_K = 6
KNN_QUERIES_PER_BUILDING = 2000
KNN_TREE_CAP = 1_500_000  # same cap DBSCAN clusters at
MEANSHIFT_QUANTILES = [0.01, 0.02, 0.05, 0.1, 0.3, 0.5]
MEANSHIFT_SAMPLES = 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    files = list_buildings(args.data_root, "train")
    assert len(files) == 18, f"expected 18 training buildings, found {len(files)}"
    rng = np.random.RandomState(SEED)

    # Pass 1: pooled per-class voxel feature statistics.
    sums = {c: np.zeros(7) for c in THING_CLASSES}
    sumsq = {c: np.zeros(7) for c in THING_CLASSES}
    counts = {c: 0 for c in THING_CLASSES}
    sizes = {c: [] for c in THING_CLASSES}
    for fp in files:
        xyz, feat, semantic, instance = load_building(fp)
        for c in THING_CLASSES:
            idx = np.flatnonzero(semantic == c)
            if len(idx) == 0:
                continue
            vfeat, p2v = voxelize(xyz[idx], feat[idx])
            sums[c] += vfeat.sum(axis=0)
            sumsq[c] += (vfeat.astype(np.float64) ** 2).sum(axis=0)
            counts[c] += len(vfeat)
            sizes[c].extend(instance_sizes_in_voxels(instance[idx], p2v).tolist())
        print(f"pass 1: {fp.split('/')[-1]}", flush=True)

    missing = [c for c in THING_CLASSES if counts[c] == 0 or len(sizes[c]) == 0]
    if missing:
        print("WARNING: no training data for classes", [CLASS_NAMES[c] for c in missing], flush=True)
    present = [c for c in THING_CLASSES if c not in missing]

    stats = {}
    for c in present:
        mu = sums[c] / counts[c]
        sigma = np.sqrt(np.maximum(sumsq[c] / counts[c] - mu ** 2, 0.0)) + 1e-6
        stats[c] = (mu, sigma)

    # Pass 2: k-distance and bandwidth in the standardised space.
    knn = {c: [] for c in THING_CLASSES}
    bandwidth = {c: {q: [] for q in MEANSHIFT_QUANTILES} for c in THING_CLASSES}
    for fp in files:
        xyz, feat, semantic, _ = load_building(fp, with_instance=False)
        for c in present:
            idx = np.flatnonzero(semantic == c)
            if len(idx) == 0:
                continue
            vfeat, _ = voxelize(xyz[idx], feat[idx])
            z = (vfeat - stats[c][0]) / stats[c][1]
            if len(z) <= KNN_K + 1:
                continue
            tree_pts = z if len(z) <= KNN_TREE_CAP else z[rng.choice(len(z), KNN_TREE_CAP, replace=False)]
            queries = tree_pts[rng.choice(len(tree_pts), min(KNN_QUERIES_PER_BUILDING, len(tree_pts)), replace=False)]
            dist, _ = cKDTree(tree_pts).query(queries, k=KNN_K + 1)  # column 0 is the point itself
            knn[c].append(float(np.median(dist[:, KNN_K])))
            sample = z[rng.choice(len(z), min(MEANSHIFT_SAMPLES, len(z)), replace=False)]
            for q in MEANSHIFT_QUANTILES:
                bw = estimate_bandwidth(sample, quantile=q, random_state=SEED)
                if bw > 0:
                    bandwidth[c][q].append(float(bw))
        print(f"pass 2: {fp.split('/')[-1]}", flush=True)

    out = dict(
        protocol_version="1.1",
        voxel_size=VOXEL_SIZE,
        knn_k=KNN_K,
        meanshift_quantiles=MEANSHIFT_QUANTILES,
        classes_without_training_data=[CLASS_NAMES[c] for c in missing],
        classes={},
    )
    for c in present:
        s = np.array(sizes[c])
        out["classes"][str(c)] = dict(
            name=CLASS_NAMES[c],
            n_train_voxels=int(counts[c]),
            n_train_instances=int(len(s)),
            feat_mean=stats[c][0].tolist(),
            feat_std=stats[c][1].tolist(),
            dbscan_eps_base=float(np.median(knn[c])),
            instance_size_p25=float(np.percentile(s, 25)),
            instance_size_median=float(np.median(s)),
            instance_size_p75=float(np.percentile(s, 75)),
            meanshift_bandwidth={str(q): float(np.median(v)) for q, v in bandwidth[c].items()},
        )
        print(c, CLASS_NAMES[c], json.dumps(out["classes"][str(c)]["meanshift_bandwidth"]),
              "eps_base", out["classes"][str(c)]["dbscan_eps_base"],
              "size p25/median", out["classes"][str(c)]["instance_size_p25"], out["classes"][str(c)]["instance_size_median"])
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
