#!/usr/bin/env python3
"""Define functional-attribute ambiguity from validated Train GT only."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class AttributeRate:
    semantic_class: str
    attribute: str
    positive_count: int
    negative_count: int
    valid_count: int
    positive_rate: float | None
    ambiguous: bool


def decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return str(value)


def h5_split(path: Path) -> str:
    """Read only the scalar split metadata, not any GT arrays."""
    with h5py.File(path, "r") as handle:
        return decode(handle["metadata/split"][()])


def collect_train(validated_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    files = sorted(validated_root.rglob("*.h5"))
    split_files: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    unexpected_splits: list[list[str]] = []
    for path in files:
        split = h5_split(path)
        if split in split_files:
            split_files[split].append(path)
        else:
            unexpected_splits.append([str(path), split])

    if unexpected_splits:
        raise ValueError(f"Unexpected split metadata: {unexpected_splits}")
    if not split_files["train"]:
        raise ValueError(f"No Train H5 files found below {validated_root}")

    attribute_order: list[str] | None = None
    classes_parts: list[np.ndarray] = []
    eval_parts: list[np.ndarray] = []
    applicable_parts: list[np.ndarray] = []
    gt_parts: list[np.ndarray] = []
    building_summaries: list[dict[str, Any]] = []

    # Only Train files pass this point. Val/Test GT arrays are never opened.
    for path in split_files["train"]:
        with h5py.File(path, "r") as handle:
            split = decode(handle["metadata/split"][()])
            if split != "train":
                raise AssertionError(f"Non-Train file reached Train loader: {path} ({split})")
            building = decode(handle["metadata/building"][()])
            attributes = [decode(item) for item in handle["metadata/attribute_names"][:]]
            if attribute_order is None:
                attribute_order = attributes
            elif attributes != attribute_order:
                raise ValueError(f"Attribute order mismatch in {path}")

            semantic_class = np.asarray(
                [decode(item) for item in handle["functional/semantic_class"][:]], dtype=object
            )
            eval_include = np.asarray(handle["functional/eval_include"][:], dtype=bool)
            applicable = np.asarray(handle["functional/applicable_mask"][:], dtype=bool)
            human_gt = np.asarray(handle["functional/attribute_human"][:])
            expected_shape = (len(semantic_class), len(attributes))
            if applicable.shape != expected_shape or human_gt.shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch in {path}: applicable={applicable.shape}, "
                    f"human_gt={human_gt.shape}, expected={expected_shape}"
                )
            if eval_include.shape != (len(semantic_class),):
                raise ValueError(f"eval_include shape mismatch in {path}: {eval_include.shape}")
            applicable_gt = human_gt[applicable]
            if not np.all(np.isin(applicable_gt, [0, 1])):
                values = np.unique(applicable_gt).tolist()
                raise ValueError(f"Non-binary applicable GT values in {path}: {values}")

            classes_parts.append(semantic_class)
            eval_parts.append(eval_include)
            applicable_parts.append(applicable)
            gt_parts.append(human_gt.astype(np.uint8, copy=False))
            building_summaries.append({
                "building": building,
                "instances": int(len(semantic_class)),
                "eval_include_true": int(eval_include.sum()),
                "eval_include_false": int((~eval_include).sum()),
            })

    assert attribute_order is not None
    data = {
        "attributes": attribute_order,
        "semantic_class": np.concatenate(classes_parts),
        "eval_include": np.concatenate(eval_parts),
        "applicable": np.concatenate(applicable_parts, axis=0),
        "human_gt": np.concatenate(gt_parts, axis=0),
    }
    provenance = {
        "all_h5_files": len(files),
        "split_file_counts": {split: len(paths) for split, paths in split_files.items()},
        "train_files_used": [str(path) for path in split_files["train"]],
        "val_test_gt_files_used": 0,
        "building_summaries": building_summaries,
    }
    return data, provenance


def compute_rates(data: dict[str, Any], tau: float) -> tuple[list[AttributeRate], dict[str, int], dict[str, Any]]:
    if not 0.0 <= tau <= 0.5:
        raise ValueError("tau must be in [0, 0.5]")
    attributes: list[str] = data["attributes"]
    semantic_class: np.ndarray = data["semantic_class"]
    eval_include: np.ndarray = data["eval_include"]
    applicable: np.ndarray = data["applicable"]
    human_gt: np.ndarray = data["human_gt"]

    class_names = sorted(set(semantic_class.tolist()))
    valid_instance_counts = {
        class_name: int(np.sum((semantic_class == class_name) & eval_include))
        for class_name in class_names
    }
    rates: list[AttributeRate] = []
    denominator_mask = eval_include[:, None] & applicable
    for class_name in class_names:
        class_mask = semantic_class == class_name
        valid_cells = class_mask[:, None] & denominator_mask
        for column, attribute in enumerate(attributes):
            cell_mask = valid_cells[:, column]
            valid_count = int(cell_mask.sum())
            positive_count = int(np.sum(human_gt[cell_mask, column] == 1))
            negative_count = valid_count - positive_count
            rate = positive_count / valid_count if valid_count else None
            ambiguous = rate is not None and tau <= rate <= 1.0 - tau
            rates.append(AttributeRate(
                semantic_class=class_name,
                attribute=attribute,
                positive_count=positive_count,
                negative_count=negative_count,
                valid_count=valid_count,
                positive_rate=rate,
                ambiguous=ambiguous,
            ))

    excluded_instance_mask = ~eval_include
    non_applicable_mask = ~applicable
    sanity = {
        "train_instances_total": int(len(eval_include)),
        "train_eval_include_true": int(eval_include.sum()),
        "train_eval_include_false": int(excluded_instance_mask.sum()),
        "eval_include_false_instances_entering_statistics": int(np.sum(
            excluded_instance_mask & np.any(denominator_mask, axis=1)
        )),
        "applicable_false_cells_total": int(non_applicable_mask.sum()),
        "applicable_false_cells_entering_denominator": int(np.sum(denominator_mask & non_applicable_mask)),
        "applicable_true_cells_from_excluded_instances_entering_denominator": int(np.sum(
            denominator_mask & excluded_instance_mask[:, None]
        )),
        "valid_cells_used": int(denominator_mask.sum()),
    }
    if sanity["eval_include_false_instances_entering_statistics"] != 0:
        raise AssertionError("eval_include=false instance entered ambiguity statistics")
    if sanity["applicable_false_cells_entering_denominator"] != 0:
        raise AssertionError("applicable=false cell entered ambiguity denominator")
    if sanity["applicable_true_cells_from_excluded_instances_entering_denominator"] != 0:
        raise AssertionError("Cell from an excluded instance entered ambiguity denominator")
    return rates, valid_instance_counts, sanity


def write_csv(path: Path, rates: list[AttributeRate]) -> None:
    fields = [
        "semantic_class", "attribute", "positive_count", "negative_count",
        "valid_count", "positive_rate", "ambiguous",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rate in rates:
            row = asdict(rate)
            row["positive_rate"] = "" if rate.positive_rate is None else f"{rate.positive_rate:.10f}"
            row["ambiguous"] = str(rate.ambiguous).lower()
            writer.writerow(row)


def render_report(
    rates: list[AttributeRate],
    valid_instance_counts: dict[str, int],
    sanity: dict[str, Any],
    provenance: dict[str, Any],
    tau: float,
) -> str:
    ambiguous_rates = [rate for rate in rates if rate.ambiguous]
    lines = [
        "Functional Attribute Ambiguity Report",
        "=====================================",
        f"Definition: Train only; eval_include=true; applicable_mask=true; tau={tau:g}",
        f"Ambiguous rule: {tau:g} <= r(c,a) <= {1.0 - tau:g}",
        f"Buildings used: {provenance['split_file_counts']['train']} Train / 0 Val / 0 Test",
        f"Ambiguous pairs: {len(ambiguous_rates)}",
        f"Ambiguous classes: {len({rate.semantic_class for rate in ambiguous_rates})}",
        "",
        "Sanity checks",
        "-------------",
        json.dumps(sanity, indent=2),
        "",
        "Per-class rates",
        "---------------",
    ]
    rates_by_class: dict[str, list[AttributeRate]] = {}
    for rate in rates:
        rates_by_class.setdefault(rate.semantic_class, []).append(rate)
    for class_name, class_rates in rates_by_class.items():
        ambiguous_attributes = [rate.attribute for rate in class_rates if rate.ambiguous]
        lines.extend([
            "",
            f"{class_name}:",
            f"  valid train instances (eval_include=true): {valid_instance_counts[class_name]}",
            "  ambiguous attributes: " + (", ".join(ambiguous_attributes) if ambiguous_attributes else "none"),
        ])
        for rate in class_rates:
            rate_text = "N/A" if rate.positive_rate is None else f"{rate.positive_rate:.6f}"
            status = "ambiguous" if rate.ambiguous else "not ambiguous"
            lines.append(
                f"  {rate.attribute:<22} {rate_text:>8}  {status:<13} "
                f"(positive={rate.positive_count}, negative={rate.negative_count}, valid={rate.valid_count})"
            )
    lines.append("")
    return "\n".join(lines)


def run(validated_root: Path, output_root: Path, tau: float) -> dict[str, Any]:
    data, provenance = collect_train(validated_root)
    rates, valid_instance_counts, sanity = compute_rates(data, tau)
    ambiguous_rates = [rate for rate in rates if rate.ambiguous]
    ambiguous_class_names = sorted({rate.semantic_class for rate in ambiguous_rates})
    by_class = {
        class_name: [
            {"attribute": rate.attribute, "positive_rate": rate.positive_rate,
             "positive_count": rate.positive_count, "negative_count": rate.negative_count,
             "valid_count": rate.valid_count}
            for rate in ambiguous_rates if rate.semantic_class == class_name
        ]
        for class_name in ambiguous_class_names
    }

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "train_attribute_rates.csv", rates)
    write_csv(output_root / "ambiguous_pairs.csv", ambiguous_rates)
    (output_root / "ambiguous_pairs.json").write_text(json.dumps({
        "definition": {"split": "train", "eval_include": True, "applicable": True, "tau": tau},
        "count": len(ambiguous_rates),
        "pairs": [asdict(rate) for rate in ambiguous_rates],
    }, indent=2) + "\n", encoding="utf-8")
    (output_root / "ambiguous_classes.json").write_text(json.dumps({
        "definition": "semantic class with at least one ambiguous Train (class, attribute) pair",
        "count": len(ambiguous_class_names),
        "ambiguous_classes": ambiguous_class_names,
        "valid_train_instance_count": valid_instance_counts,
        "ambiguous_attributes_by_class": by_class,
    }, indent=2) + "\n", encoding="utf-8")
    report_text = render_report(rates, valid_instance_counts, sanity, provenance, tau)
    (output_root / "ambiguity_report.txt").write_text(report_text, encoding="utf-8")
    print(report_text, end="")
    return {
        "rates": rates,
        "ambiguous_pairs": ambiguous_rates,
        "ambiguous_classes": ambiguous_class_names,
        "valid_instance_counts": valid_instance_counts,
        "sanity": sanity,
        "provenance": provenance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validated-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args.validated_root, args.output_root, args.tau)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
