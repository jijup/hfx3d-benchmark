#!/usr/bin/env python3
"""Evaluate functional assignment CSVs against validated H5 labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from build_ablation_scores import (
    SETTING_FLAGS,
    build_ablation_scores,
    load_evaluation_split,
    load_frozen_ambiguous_classes,
)
from evaluate_ablation import PER_ATTRIBUTE_FIELDS, evaluate_split, write_rows as write_attribute_rows


THRESHOLD = 0.5


def binary_f1(truth: np.ndarray, score: np.ndarray) -> float:
    prediction = score >= THRESHOLD
    target = truth.astype(bool, copy=False)
    tp = int(np.sum(prediction & target))
    fp = int(np.sum(prediction & ~target))
    fn = int(np.sum(~prediction & target))
    denominator = 2 * tp + fp + fn
    return 2.0 * tp / denominator if denominator else 0.0


def evaluate_mf1_esm(data: Any, scores: dict[str, np.ndarray], ambiguous_classes: set[str]):
    subsets = {
        "All": data.eval_include,
        "Ambiguous": data.eval_include & np.isin(data.semantic_class, sorted(ambiguous_classes)),
    }
    metrics: dict[str, dict[str, tuple[float, int, int]]] = {}
    for subset, instance_mask in subsets.items():
        valid_cells = instance_mask[:, None] & data.applicable
        eligible = []
        for column in range(len(data.attributes)):
            truth = data.human_gt[valid_cells[:, column], column]
            if len(truth) and np.any(truth == 0) and np.any(truth == 1):
                eligible.append(column)
        if not eligible:
            raise ValueError(f"{data.split}/{subset}: no metric-eligible attributes")
        metrics[subset] = {}
        for setting, matrix in scores.items():
            mf1 = float(np.mean([
                binary_f1(
                    data.human_gt[valid_cells[:, column], column],
                    matrix[valid_cells[:, column], column],
                )
                for column in eligible
            ]))
            prediction = matrix >= THRESHOLD
            exact = np.all(
                (prediction == data.human_gt.astype(bool)) | ~data.applicable,
                axis=1,
            )
            exact_count = int(np.sum(exact & instance_mask))
            count = int(instance_mask.sum())
            metrics[subset][setting] = (mf1, exact_count, count)

    rows = []
    for setting in SETTING_FLAGS:
        all_f1, all_exact, all_count = metrics["All"][setting]
        amb_f1, amb_exact, amb_count = metrics["Ambiguous"][setting]
        rows.append({
            "Evidence": setting,
            "All mF1": all_f1,
            "All ESM": all_exact / all_count,
            "Ambiguous mF1": amb_f1,
            "Ambiguous ESM": amb_exact / amb_count,
            "All exact": all_exact,
            "All n": all_count,
            "Ambiguous exact": amb_exact,
            "Ambiguous n": amb_count,
        })
    return rows


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "Evidence", "All mF1", "All ESM", "Ambiguous mF1", "Ambiguous ESM",
        "All exact", "All n", "Ambiguous exact", "Ambiguous n",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for source in rows:
            row = dict(source)
            for field in ("All mF1", "All ESM", "Ambiguous mF1", "Ambiguous ESM"):
                row[field] = f"{row[field]:.10f}"
            writer.writerow(row)


def latex_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[t]", r"\centering", r"\begin{tabular}{lcccc}", r"\toprule",
        r"Evidence & All mF1 & All ESM & Ambiguous mF1 & Ambiguous ESM \\", r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['Evidence']} & {row['All mF1']:.4f} & {row['All ESM']:.4f} & "
            f"{row['Ambiguous mF1']:.4f} & {row['Ambiguous ESM']:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def run(
    prediction_csv_root: Path,
    validated_h5_root: Path,
    ambiguous_classes_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    ambiguous_classes, ambiguous_payload = load_frozen_ambiguous_classes(ambiguous_classes_path)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    audits = {}
    for split in ("train", "val", "test"):
        data = load_evaluation_split(prediction_csv_root, validated_h5_root, split)
        scores, consistency = build_ablation_scores(data)
        rows = evaluate_mf1_esm(data, scores, ambiguous_classes)
        summaries[split] = rows
        write_summary(output_root / f"{split}_ablation_mf1_esm.csv", rows)
        _, attribute_rows, subset_audit = evaluate_split(data, scores, ambiguous_classes)
        audits[split] = {
            "building_count": data.building_count,
            "instances_total": int(len(data.instance_id)),
            "eval_include_true": int(data.eval_include.sum()),
            "eval_include_false": int((~data.eval_include).sum()),
            "source_consistency": consistency,
            "subsets": subset_audit,
        }
        if split == "test":
            for subset, filename in (
                ("All", "per_attribute_all.csv"),
                ("Ambiguous", "per_attribute_ambiguous.csv"),
            ):
                selected = [row for row in attribute_rows if row["subset"] == subset]
                write_attribute_rows(output_root / filename, selected, PER_ATTRIBUTE_FIELDS)

    (output_root / "ablation_table.tex").write_text(
        latex_table(summaries["test"]), encoding="utf-8"
    )
    audit = {
        "prediction_input": str(prediction_csv_root),
        "prediction_fields": [
            "semantic_prior_score", "geometry_score", "context_score", "final_confidence",
        ],
        "human_validation_input": str(validated_h5_root),
        "ground_truth_field": "functional/attribute_human",
        "mask_fields": ["functional/eval_include", "functional/applicable_mask"],
        "ambiguous_classes_input": str(ambiguous_classes_path),
        "ambiguous_class_count": ambiguous_payload["count"],
        "threshold": THRESHOLD,
        "splits": audits,
    }
    (output_root / "evaluation_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"test_rows": summaries["test"], "output_root": str(output_root)}, indent=2))
    return {"summaries": summaries, "audit": audit}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-csv-root", type=Path, required=True,
        help="Root containing functional assignment CSV files, preferably train/val/test subdirectories",
    )
    parser.add_argument(
        "--validated-h5-root", type=Path, required=True,
        help="Root containing final human-validation Train/Val/Test H5 files",
    )
    parser.add_argument(
        "--ambiguous-classes", type=Path, required=True,
        help="Train-only ambiguous_classes.json produced by compute_ambiguity.py",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(
        args.prediction_csv_root,
        args.validated_h5_root,
        args.ambiguous_classes,
        args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
