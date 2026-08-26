#!/usr/bin/env python3
"""Load assignment CSV scores and validated H5 labels for evaluation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SETTING_FLAGS: dict[str, tuple[int, int, int]] = {
    "S only": (1, 0, 0),
    "G only": (0, 1, 0),
    "C only": (0, 0, 1),
    "S+G": (1, 1, 0),
    "S+C": (1, 0, 1),
    "S+G+C": (1, 1, 1),
}

CSV_REQUIRED_FIELDS = {
    "building_id", "instance_id", "semantic_class", "attribute_name",
    "applicable", "semantic_prior_score", "geometry_score", "context_score",
    "final_confidence",
}


@dataclass(frozen=True)
class SplitData:
    split: str
    attributes: tuple[str, ...]
    building_id: np.ndarray
    instance_id: np.ndarray
    semantic_class: np.ndarray
    eval_include: np.ndarray
    applicable: np.ndarray
    human_gt: np.ndarray
    semantic_prior: np.ndarray
    geometry_score: np.ndarray
    context_score: np.ndarray
    confidence: np.ndarray
    building_count: int


def decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return str(value)


def parse_bool(value: str, field: str, path: Path, line: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"{path}:{line}: invalid {field} value {value!r}")


def odds_fusion(
    semantic_prior: np.ndarray,
    geometry_score: np.ndarray,
    context_score: np.ndarray,
    odds_epsilon: float = 1e-6,
) -> np.ndarray:
    """Vectorized equivalent of the assignment pipeline's odds fusion."""
    if not 0.0 < odds_epsilon < 0.5:
        raise ValueError("odds_epsilon must be in (0, 0.5)")
    values = [
        np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
        for value in (semantic_prior, geometry_score, context_score)
    ]
    if not (values[0].shape == values[1].shape == values[2].shape):
        raise ValueError("All odds-fusion inputs must have the same shape")
    positive = values[0] * values[1] * values[2]
    negative = (1.0 - values[0]) * (1.0 - values[1]) * (1.0 - values[2])
    denominator = positive + negative
    result = np.empty_like(denominator)
    defined = denominator > 0.0
    result[defined] = positive[defined] / denominator[defined]
    if np.any(~defined):
        clipped = [np.clip(value[~defined], odds_epsilon, 1.0 - odds_epsilon) for value in values]
        log_odds = sum(np.log(value) - np.log1p(-value) for value in clipped)
        nonnegative = log_odds >= 0.0
        boundary = np.empty_like(log_odds)
        boundary[nonnegative] = 1.0 / (1.0 + np.exp(-log_odds[nonnegative]))
        exponential = np.exp(log_odds[~nonnegative])
        boundary[~nonnegative] = exponential / (1.0 + exponential)
        result[~defined] = boundary
    return np.clip(result, 0.0, 1.0)


def _validated_files(validated_h5_root: Path, split: str) -> list[Path]:
    paths = []
    for path in sorted(validated_h5_root.rglob("*.h5")):
        with h5py.File(path, "r") as handle:
            current_split = decode(handle["metadata/split"][()]).lower()
        if current_split == split:
            paths.append(path)
    if not paths:
        raise ValueError(f"No human-validation H5 files for split={split!r} under {validated_h5_root}")
    return paths


def _load_validated_gt(validated_h5_root: Path, split: str) -> tuple[dict[str, Any], int]:
    attribute_order: tuple[str, ...] | None = None
    parts: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "building_id", "instance_id", "semantic_class", "eval_include",
            "applicable", "human_gt",
        )
    }
    paths = _validated_files(validated_h5_root, split)
    for path in paths:
        with h5py.File(path, "r") as handle:
            building = decode(handle["metadata/building"][()])
            attributes = tuple(decode(value) for value in handle["metadata/attribute_names"][:])
            if attribute_order is None:
                attribute_order = attributes
            elif attributes != attribute_order:
                raise ValueError(f"Attribute order mismatch in {path}")
            group = handle["functional"]
            instance_id = np.asarray(group["instance_id"][:], dtype=np.int64)
            semantic_class = np.asarray([decode(value) for value in group["semantic_class"][:]], dtype=object)
            eval_include = np.asarray(group["eval_include"][:], dtype=bool)
            applicable = np.asarray(group["applicable_mask"][:], dtype=bool)
            human_gt = np.asarray(group["attribute_human"][:], dtype=np.uint8)
            expected = (len(instance_id), len(attributes))
            if semantic_class.shape != (len(instance_id),) or eval_include.shape != (len(instance_id),):
                raise ValueError(f"Invalid instance vector shape in {path}")
            if applicable.shape != expected or human_gt.shape != expected:
                raise ValueError(f"Invalid functional matrix shape in {path}")
            if not np.all(np.isin(human_gt[applicable], [0, 1])):
                raise ValueError(f"Applicable human GT is not binary in {path}")
            parts["building_id"].append(np.full(len(instance_id), building, dtype=object))
            parts["instance_id"].append(instance_id)
            parts["semantic_class"].append(semantic_class)
            parts["eval_include"].append(eval_include)
            parts["applicable"].append(applicable)
            parts["human_gt"].append(human_gt)
    assert attribute_order is not None
    return {
        "attributes": attribute_order,
        **{name: np.concatenate(values, axis=0) for name, values in parts.items()},
    }, len(paths)


def _prediction_csvs(prediction_csv_root: Path, split: str) -> list[Path]:
    split_root = prediction_csv_root / split
    paths = sorted(split_root.rglob("*.csv")) if split_root.is_dir() else sorted(prediction_csv_root.rglob("*.csv"))
    if not paths:
        raise ValueError(f"No functional-assignment CSV files found for split={split!r}")
    return paths


def load_evaluation_split(
    prediction_csv_root: Path,
    validated_h5_root: Path,
    split: str,
) -> SplitData:
    """Join CSV predictions to validated H5 rows by building and instance ID."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    gt, building_count = _load_validated_gt(validated_h5_root, split)
    attributes: tuple[str, ...] = gt["attributes"]
    attribute_index = {name: index for index, name in enumerate(attributes)}
    row_index = {
        (str(building), int(instance)): index
        for index, (building, instance) in enumerate(zip(gt["building_id"], gt["instance_id"]))
    }
    if len(row_index) != len(gt["instance_id"]):
        raise ValueError(f"Duplicate validated building/instance key in split={split}")

    shape = gt["applicable"].shape
    matrices = {
        name: np.full(shape, np.nan, dtype=np.float64)
        for name in ("semantic_prior", "geometry_score", "context_score", "confidence")
    }
    seen = np.zeros(shape, dtype=bool)
    csv_buildings: set[str] = set()
    for path in _prediction_csvs(prediction_csv_root, split):
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            missing = CSV_REQUIRED_FIELDS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: missing assignment CSV columns {sorted(missing)}")
            for line, record in enumerate(reader, 2):
                building = record["building_id"].strip()
                key = (building, int(record["instance_id"]))
                if key not in row_index:
                    continue
                csv_buildings.add(building)
                row = row_index[key]
                if record["semantic_class"].strip() != str(gt["semantic_class"][row]):
                    raise ValueError(f"{path}:{line}: semantic_class disagrees with validated H5")
                attribute = record["attribute_name"].strip()
                if attribute not in attribute_index:
                    raise ValueError(f"{path}:{line}: unknown attribute {attribute!r}")
                column = attribute_index[attribute]
                csv_applicable = parse_bool(record["applicable"], "applicable", path, line)
                h5_applicable = bool(gt["applicable"][row, column])
                if csv_applicable != h5_applicable:
                    raise ValueError(f"{path}:{line}: applicable disagrees with validated H5")
                if not csv_applicable:
                    continue
                if seen[row, column]:
                    raise ValueError(f"{path}:{line}: duplicate applicable prediction cell")
                values = {
                    "semantic_prior": float(record["semantic_prior_score"]),
                    "geometry_score": float(record["geometry_score"]),
                    "context_score": float(record["context_score"]),
                    "confidence": float(record["final_confidence"]),
                }
                if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in values.values()):
                    raise ValueError(f"{path}:{line}: prediction score outside [0,1]")
                for name, value in values.items():
                    matrices[name][row, column] = value
                seen[row, column] = True

    missing_cells = gt["applicable"] & ~seen
    if np.any(missing_cells):
        examples = []
        for row, column in np.argwhere(missing_cells)[:10]:
            examples.append((str(gt["building_id"][row]), int(gt["instance_id"][row]), attributes[column]))
        raise ValueError(f"Missing {int(missing_cells.sum())} applicable CSV cells; examples={examples}")
    expected_buildings = set(str(value) for value in gt["building_id"])
    if csv_buildings != expected_buildings:
        raise ValueError(
            f"CSV/H5 building mismatch for {split}: missing={sorted(expected_buildings-csv_buildings)}, "
            f"unexpected={sorted(csv_buildings-expected_buildings)}"
        )
    for matrix in matrices.values():
        matrix[~gt["applicable"]] = 0.0
    return SplitData(split=split, building_count=building_count, **gt, **matrices)


def build_ablation_scores(data: SplitData, odds_epsilon: float = 1e-6) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    neutral = np.full_like(data.semantic_prior, 0.5, dtype=np.float64)
    scores = {
        "S only": data.semantic_prior.copy(),
        "G only": data.geometry_score.copy(),
        "C only": data.context_score.copy(),
        "S+G": odds_fusion(data.semantic_prior, data.geometry_score, neutral, odds_epsilon),
        "S+C": odds_fusion(data.semantic_prior, neutral, data.context_score, odds_epsilon),
        "S+G+C": data.confidence.copy(),
    }
    recomputed = odds_fusion(data.semantic_prior, data.geometry_score, data.context_score, odds_epsilon)
    errors = np.abs(recomputed[data.applicable] - data.confidence[data.applicable])
    consistency = {
        "split": data.split,
        "applicable_cells_checked": int(errors.size),
        "max_absolute_error": float(errors.max(initial=0.0)),
        "mean_absolute_error": float(errors.mean()) if errors.size else 0.0,
        "mismatch_count_at_1e-6": int(np.sum(errors > 1e-6)),
        "odds_epsilon": odds_epsilon,
        "prediction_source": "functional assignment CSV",
        "ground_truth_source": "human-validation H5",
    }
    if consistency["mismatch_count_at_1e-6"]:
        raise ValueError(f"CSV final_confidence fails odds-fusion consistency: {consistency}")
    return scores, consistency


def load_frozen_ambiguous_classes(path: Path) -> tuple[set[str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    classes = payload.get("ambiguous_classes")
    if not isinstance(classes, list) or not classes or not all(isinstance(value, str) for value in classes):
        raise ValueError(f"Invalid frozen ambiguous class file: {path}")
    if payload.get("count") != len(classes):
        raise ValueError(f"Frozen ambiguous class count mismatch: {path}")
    return set(classes), payload
