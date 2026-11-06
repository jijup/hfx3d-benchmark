#!/usr/bin/env python3
"""Per-attribute F1/AP and macro metric utilities for functional evaluation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from build_ablation_scores import SETTING_FLAGS, SplitData


THRESHOLD = 0.5


def binary_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    predicted = scores >= THRESHOLD
    truth = y_true.astype(bool, copy=False)
    true_positive = int(np.sum(predicted & truth))
    false_positive = int(np.sum(predicted & ~truth))
    false_negative = int(np.sum(~predicted & truth))
    denominator = 2 * true_positive + false_positive + false_negative
    return (2.0 * true_positive / denominator) if denominator else 0.0


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Non-interpolated AP with tied confidence values grouped exactly."""
    truth = y_true.astype(np.uint8, copy=False)
    positive_count = int(truth.sum())
    negative_count = len(truth) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AP requires at least one positive and one negative GT")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_truth = truth[order]
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)) + 1, len(sorted_scores)]
    cumulative_tp = np.cumsum(sorted_truth, dtype=np.int64)
    tp_at_end = cumulative_tp[group_ends - 1]
    precision = tp_at_end / group_ends
    positives_per_group = np.diff(np.r_[0, tp_at_end])
    return float(np.sum((positives_per_group / positive_count) * precision))


def evaluate_split(
    data: SplitData,
    scores_by_setting: dict[str, np.ndarray],
    ambiguous_classes: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    masks = {
        "All": data.eval_include.copy(),
        "Ambiguous": data.eval_include & np.isin(data.semantic_class, sorted(ambiguous_classes)),
    }
    per_attribute_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    subset_audit: dict[str, Any] = {}

    for subset_name, instance_mask in masks.items():
        valid_cell_mask = instance_mask[:, None] & data.applicable
        if np.any((~data.eval_include)[:, None] & valid_cell_mask):
            raise AssertionError(f"{data.split}/{subset_name}: eval_include=false entered scoring")
        if np.any((~data.applicable) & valid_cell_mask):
            raise AssertionError(f"{data.split}/{subset_name}: applicable=false entered scoring")

        eligible_attributes: list[str] = []
        excluded_attributes: dict[str, str] = {}
        for column, attribute in enumerate(data.attributes):
            truth = data.human_gt[valid_cell_mask[:, column], column]
            positives = int(truth.sum())
            negatives = len(truth) - positives
            if len(truth) == 0:
                excluded_attributes[attribute] = "no valid applicable cells"
            elif positives == 0:
                excluded_attributes[attribute] = "no positive GT"
            elif negatives == 0:
                excluded_attributes[attribute] = "no negative GT"
            else:
                eligible_attributes.append(attribute)

        setting_metrics: dict[str, dict[str, float]] = {}
        for setting, score_matrix in scores_by_setting.items():
            f1_values: list[float] = []
            ap_values: list[float] = []
            for column, attribute in enumerate(data.attributes):
                cell_mask = valid_cell_mask[:, column]
                truth = data.human_gt[cell_mask, column]
                scores = score_matrix[cell_mask, column]
                positives = int(truth.sum())
                negatives = len(truth) - positives
                eligible = attribute in eligible_attributes
                f1 = binary_f1(truth, scores) if eligible else None
                ap = average_precision(truth, scores) if eligible else None
                if eligible:
                    assert f1 is not None and ap is not None
                    f1_values.append(f1)
                    ap_values.append(ap)
                s_flag, g_flag, c_flag = SETTING_FLAGS[setting]
                per_attribute_rows.append({
                    "split": data.split,
                    "subset": subset_name,
                    "Setting": setting,
                    "S": s_flag,
                    "G": g_flag,
                    "C": c_flag,
                    "attribute": attribute,
                    "F1": f1,
                    "AP": ap,
                    "positive_gt_count": positives,
                    "negative_gt_count": negatives,
                    "valid_count": int(len(truth)),
                    "included_in_macro": eligible,
                    "exclusion_reason": "" if eligible else excluded_attributes[attribute],
                })
            if not f1_values:
                raise ValueError(f"{data.split}/{subset_name}: no metric-eligible attributes")
            setting_metrics[setting] = {
                "Macro_F1": float(np.mean(f1_values)),
                "mAP": float(np.mean(ap_values)),
            }

        for setting in SETTING_FLAGS:
            s_flag, g_flag, c_flag = SETTING_FLAGS[setting]
            existing = next((row for row in summary_rows if row["Setting"] == setting), None)
            if existing is None:
                existing = {"Setting": setting, "S": s_flag, "G": g_flag, "C": c_flag}
                summary_rows.append(existing)
            existing[f"{subset_name}_Macro_F1"] = setting_metrics[setting]["Macro_F1"]
            existing[f"{subset_name}_mAP"] = setting_metrics[setting]["mAP"]

        subset_audit[subset_name] = {
            "instances": int(instance_mask.sum()),
            "valid_applicable_cells": int(valid_cell_mask.sum()),
            "eligible_attributes": eligible_attributes,
            "excluded_attributes": excluded_attributes,
            "eval_include_false_instances_entering_scoring": int(
                np.sum((~data.eval_include) & instance_mask)
            ),
            "applicable_false_cells_entering_scoring": 0,
        }
    return summary_rows, per_attribute_rows, subset_audit


PER_ATTRIBUTE_FIELDS = [
    "split", "subset", "Setting", "S", "G", "C", "attribute", "F1", "AP",
    "positive_gt_count", "negative_gt_count", "valid_count", "included_in_macro",
    "exclusion_reason",
]


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for input_row in rows:
            row = dict(input_row)
            for field in ("F1", "AP"):
                if field in row:
                    row[field] = "" if row[field] is None else f"{row[field]:.10f}"
            if "included_in_macro" in row:
                row["included_in_macro"] = str(row["included_in_macro"]).lower()
            writer.writerow(row)
