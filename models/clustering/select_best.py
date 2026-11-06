"""
Pick each clustering method's scale parameter on the VAL split: highest val mAP, ties broken by AP50, then by the smaller grid
index. Prints the chosen grid index (for the test job) and records the choice.

Usage:
    python models/clustering/select_best.py --method dbscan --results-dir results/clustering
"""

import argparse
import glob
import json
import os

from eval_cluster_hfx3d import GRIDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(GRIDS), required=True)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    rows = []
    for fp in glob.glob(os.path.join(args.results_dir, f"{args.method}_g*_val.json")):
        with open(fp) as f:
            r = json.load(f)
        rows.append((r["mAP"], r["AP50"], -r["grid_index"], r))
    missing = sorted(set(range(len(GRIDS[args.method]))) - {r[3]["grid_index"] for r in rows})
    if not rows:
        raise SystemExit(f"no val results for {args.method}")
    best = max(rows)[3]
    choice = dict(
        method=args.method, chosen_grid_index=best["grid_index"], chosen_value=best["value"],
        val_mAP=best["mAP"], val_AP50=best["AP50"], missing_grid_indices=missing,
        val_sweep=sorted(
            [dict(grid_index=r[3]["grid_index"], value=r[3]["value"], mAP=r[3]["mAP"], AP50=r[3]["AP50"]) for r in rows],
            key=lambda d: d["grid_index"],
        ),
    )
    with open(os.path.join(args.results_dir, f"{args.method}_selection.json"), "w") as f:
        json.dump(choice, f, indent=2)
    print(best["grid_index"])


if __name__ == "__main__":
    main()
