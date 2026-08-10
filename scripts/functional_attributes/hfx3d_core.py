"""HFX3D refinement core: I/O, ontology, schema, instance extraction, writers, validation.

Independent re-implementation of the base I/O layer for the simple
Geometry-Guided Functional Refinement pipeline. No scoring logic lives here.
"""
from __future__ import annotations

__version__ = "1.0.0"

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import yaml


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def clip01(value: float | np.ndarray) -> float | np.ndarray:
    return np.clip(value, 0.0, 1.0)


def deterministic_direction(vector: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64).copy()
    norm = float(np.linalg.norm(result))
    if norm <= epsilon:
        return result
    result /= norm
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0:
        result *= -1.0
    return result


@dataclass(frozen=True)
class LabelMap:
    semantic_id_to_name: dict[int, str]
    semantic_name_to_id: dict[str, int]


def _is_int_like(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, str):
        try:
            int(value)
            return str(int(value)) == value.strip()
        except ValueError:
            return False
    return False


def load_label_map(path: str | Path) -> LabelMap:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("label_map.json must contain a non-empty object")

    keys_int = all(_is_int_like(key) for key in raw)
    values_int = all(_is_int_like(value) for value in raw.values())
    keys_str = all(isinstance(key, str) for key in raw)
    values_str = all(isinstance(value, str) for value in raw.values())

    if keys_int and values_str:
        id_to_name = {int(key): value for key, value in raw.items()}
    elif keys_str and values_int:
        id_to_name = {int(value): key for key, value in raw.items()}
    else:
        raise ValueError(
            "label_map.json must be entirely ID->name or entirely name->ID; mixed/ambiguous maps are invalid"
        )

    if any(semantic_id < 0 for semantic_id in id_to_name):
        raise ValueError("label_map.json contains a negative semantic ID")
    if any(not name for name in id_to_name.values()):
        raise ValueError("label_map.json contains an empty semantic class name")
    if len(set(id_to_name.values())) != len(id_to_name):
        raise ValueError("label_map.json contains duplicate semantic class names")
    name_to_id = {name: semantic_id for semantic_id, name in id_to_name.items()}
    return LabelMap(dict(sorted(id_to_name.items())), name_to_id)


@dataclass(frozen=True)
class DatasetInfo:
    path: str
    shape: tuple[int, ...]
    dtype: str
    compression: str | None
    chunks: tuple[int, ...] | None
    attributes: dict[str, str]


@dataclass(frozen=True)
class H5Schema:
    file_path: Path
    xyz_dataset: str
    semantic_dataset: str
    instance_dataset: str
    point_count: int
    datasets: tuple[DatasetInfo, ...]


def _score_candidate(path: str, tokens: tuple[str, ...]) -> int:
    lowered = path.lower()
    components = lowered.split("/")
    score = 0
    for rank, token in enumerate(tokens):
        if components[-1] == token:
            score += 100 - rank
        elif token in components[-1]:
            score += 30 - rank
        elif token in lowered:
            score += 10 - rank
    return score


def _choose(candidates: list[str], role: str, tokens: tuple[str, ...]) -> str:
    if not candidates:
        raise ValueError(f"No shape-compatible {role} dataset found")
    scored = sorted(((_score_candidate(path, tokens), path) for path in candidates), reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        raise ValueError(f"Ambiguous {role} datasets: {scored[0][1]!r}, {scored[1][1]!r}")
    if scored[0][0] <= 0:
        raise ValueError(f"Cannot identify {role} dataset from names: {candidates}")
    return scored[0][1]


def discover_schema(path: str | Path) -> H5Schema:
    file_path = Path(path).resolve()
    datasets: list[DatasetInfo] = []
    xyz_candidates: list[str] = []
    label_candidates: list[str] = []
    with h5py.File(file_path, "r") as handle:
        def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            info = DatasetInfo(
                path=name,
                shape=tuple(int(v) for v in obj.shape),
                dtype=str(obj.dtype),
                compression=obj.compression,
                chunks=None if obj.chunks is None else tuple(int(v) for v in obj.chunks),
                attributes={str(key): repr(value) for key, value in obj.attrs.items()},
            )
            datasets.append(info)
            if len(obj.shape) == 2 and obj.shape[1] == 3 and np.issubdtype(obj.dtype, np.floating):
                xyz_candidates.append(name)
            if len(obj.shape) == 1 and np.issubdtype(obj.dtype, np.integer):
                label_candidates.append(name)
        handle.visititems(visitor)

    xyz = _choose(xyz_candidates, "XYZ", ("xyz", "coordinates", "coords", "points"))
    semantic = _choose(label_candidates, "semantic", ("semantic", "semantics", "class", "label"))
    remaining = [candidate for candidate in label_candidates if candidate != semantic]
    instance = _choose(remaining, "instance", ("instance", "instances", "object", "segment"))
    shape_by_path = {item.path: item.shape for item in datasets}
    point_count = shape_by_path[xyz][0]
    if shape_by_path[xyz] != (point_count, 3):
        raise ValueError(f"XYZ shape is invalid in {file_path}: {shape_by_path[xyz]}")
    if shape_by_path[semantic] != (point_count,) or shape_by_path[instance] != (point_count,):
        raise ValueError(
            f"Point/label length mismatch in {file_path}: xyz={shape_by_path[xyz]}, "
            f"semantic={shape_by_path[semantic]}, instance={shape_by_path[instance]}"
        )
    return H5Schema(file_path, xyz, semantic, instance, point_count, tuple(datasets))


class H5PointReader:
    def __init__(self, path: str | Path, chunk_points: int = 500_000):
        self.path = Path(path).resolve()
        self.schema: H5Schema = discover_schema(self.path)
        self.chunk_points = int(chunk_points)
        if self.chunk_points <= 0:
            raise ValueError("chunk_points must be positive")

    def iter_labels(self) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        with h5py.File(self.path, "r") as handle:
            semantic = handle[self.schema.semantic_dataset]
            instance = handle[self.schema.instance_dataset]
            for start in range(0, self.schema.point_count, self.chunk_points):
                stop = min(start + self.chunk_points, self.schema.point_count)
                yield start, np.asarray(semantic[start:stop], dtype=np.int64), np.asarray(instance[start:stop], dtype=np.int64)

    def iter_points(self) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
        with h5py.File(self.path, "r") as handle:
            xyz = handle[self.schema.xyz_dataset]
            semantic = handle[self.schema.semantic_dataset]
            instance = handle[self.schema.instance_dataset]
            for start in range(0, self.schema.point_count, self.chunk_points):
                stop = min(start + self.chunk_points, self.schema.point_count)
                yield (
                    start,
                    np.asarray(xyz[start:stop], dtype=np.float64),
                    np.asarray(semantic[start:stop], dtype=np.int64),
                    np.asarray(instance[start:stop], dtype=np.int64),
                )


CANONICAL_ATTRIBUTE_ORDER: tuple[str, ...] = (
    "load_bearing",
    "thermal_envelope",
    "vegetation_support",
    "operable",
    "solar_shading",
    "ventilation",
    "natural_lighting",
    "access",
    "drainage",
    "fall_protection",
    "aesthetic",
    "privacy_screening",
    "circulation",
    "illumination",
    "surveillance",
)


@dataclass(frozen=True)
class OntologyRule:
    semantic_class: str
    ifc_anchor: str
    attribute: str
    value: bool
    condition: str
    source: str
    rule_index: int


@dataclass(frozen=True)
class Ontology:
    path: Path
    sha256: str
    version: Any
    semantic_classes: tuple[str, ...]
    rules_by_class: dict[str, tuple[OntologyRule, ...]]
    rules_by_class_attribute: dict[str, dict[str, tuple[OntologyRule, ...]]]
    attribute_names: tuple[str, ...]
    attribute_index: dict[str, int]
    unique_conditions: tuple[str, ...]


def _order_attributes(seen: list[str]) -> tuple[str, ...]:
    unknown = [name for name in seen if name not in CANONICAL_ATTRIBUTE_ORDER]
    if unknown:
        raise ValueError(f"Ontology contains attributes outside the fixed 15-dim schema: {unknown}")
    missing = [name for name in CANONICAL_ATTRIBUTE_ORDER if name not in seen]
    if missing:
        raise ValueError(f"Ontology is missing attributes required by the fixed 15-dim schema: {missing}")
    ordered = [name for name in CANONICAL_ATTRIBUTE_ORDER if name in seen]
    return tuple(ordered)


def load_ontology(path: str | Path) -> Ontology:
    ontology_path = Path(path).resolve()
    raw = load_yaml(ontology_path)
    if "version" not in raw:
        raise ValueError("Ontology is missing version")
    classes: list[str] = []
    rules_by_class: dict[str, tuple[OntologyRule, ...]] = {}
    rules_by_class_attribute: dict[str, dict[str, tuple[OntologyRule, ...]]] = {}
    attributes_seen: list[str] = []
    conditions: list[str] = []
    global_rule_index = 0
    for semantic_class, block in raw.items():
        if semantic_class == "version":
            continue
        if not isinstance(semantic_class, str) or not isinstance(block, dict):
            raise ValueError(f"Invalid ontology class block: {semantic_class!r}")
        class_ifc_anchor = block.get("ifc_anchor")
        raw_rules = block.get("rules")
        if not isinstance(class_ifc_anchor, str) or not isinstance(raw_rules, list):
            raise ValueError(f"Class {semantic_class!r} must define string ifc_anchor and list rules")
        classes.append(semantic_class)
        parsed: list[OntologyRule] = []
        by_attribute: dict[str, list[OntologyRule]] = defaultdict(list)
        for raw_rule in raw_rules:
            if not isinstance(raw_rule, dict):
                raise ValueError(f"Rule in {semantic_class!r} must be a mapping")
            required = {"attribute", "value", "condition", "source"}
            optional = {"ifc_anchor"}
            keys = set(raw_rule)
            if not required.issubset(keys) or (keys - required - optional):
                raise ValueError(
                    f"Rule in {semantic_class!r} must have keys {sorted(required)} "
                    f"and optional {sorted(optional)}; got {sorted(keys)}"
                )
            if not isinstance(raw_rule["attribute"], str) or not isinstance(raw_rule["condition"], str):
                raise ValueError(f"Rule names must be strings in {semantic_class!r}")
            if not isinstance(raw_rule["value"], bool) or not isinstance(raw_rule["source"], str):
                raise ValueError(f"Rule value/source type invalid in {semantic_class!r}")
            if "TBD" in raw_rule["condition"]:
                raise ValueError(f"Unresolved TBD condition in {semantic_class!r}: {raw_rule['condition']!r}")
            rule_ifc = raw_rule.get("ifc_anchor", class_ifc_anchor)
            if not isinstance(rule_ifc, str):
                raise ValueError(f"Rule ifc_anchor must be string in {semantic_class!r}")
            rule = OntologyRule(
                semantic_class=semantic_class,
                ifc_anchor=rule_ifc,
                attribute=raw_rule["attribute"],
                value=raw_rule["value"],
                condition=raw_rule["condition"],
                source=raw_rule["source"],
                rule_index=global_rule_index,
            )
            global_rule_index += 1
            parsed.append(rule)
            by_attribute[rule.attribute].append(rule)
            if rule.attribute not in attributes_seen:
                attributes_seen.append(rule.attribute)
            if rule.condition not in conditions:
                conditions.append(rule.condition)
        rules_by_class[semantic_class] = tuple(parsed)
        rules_by_class_attribute[semantic_class] = {
            attribute: tuple(rules) for attribute, rules in by_attribute.items()
        }
    attribute_names = _order_attributes(attributes_seen)
    return Ontology(
        path=ontology_path,
        sha256=sha256_file(ontology_path),
        version=raw["version"],
        semantic_classes=tuple(classes),
        rules_by_class=rules_by_class,
        rules_by_class_attribute=rules_by_class_attribute,
        attribute_names=attribute_names,
        attribute_index={name: index for index, name in enumerate(attribute_names)},
        unique_conditions=tuple(conditions),
    )


@dataclass
class InstanceRecord:
    instance_id: int
    semantic_id: int
    semantic_class: str
    semantic_purity: float
    point_count: int
    centroid: np.ndarray
    xyz_min: np.ndarray
    xyz_max: np.ndarray
    sample_points: np.ndarray
    semantic_tie: bool


@dataclass
class BuildingInstances:
    source_path: Path
    records: list[InstanceRecord]
    stats: dict[str, Any]
    semantic_point_counts: dict[int, int]
    semantic_instance_counts: dict[int, int]
    schema: Any


def extract_instances(
    path: str | Path,
    label_map: LabelMap,
    config: dict[str, Any],
) -> BuildingInstances:
    io_config = config["io"]
    sampling = config["sampling"]
    reader = H5PointReader(path, chunk_points=int(io_config["chunk_points"]))
    known_ids = np.asarray(sorted(label_map.semantic_id_to_name), dtype=np.int64)
    max_semantic_id = int(known_ids.max())
    counts_by_instance: dict[int, int] = {}
    semantic_histograms: dict[int, np.ndarray] = {}
    semantic_point_counts = {int(key): 0 for key in known_ids}
    raw_instance_ids: set[int] = set()
    semantic_minus1_count = 0
    negative_instance_count = 0
    valid_point_count = 0

    for _, semantic, instance in reader.iter_labels():
        semantic_minus1_count += int(np.count_nonzero(semantic == -1))
        negative_instance_count += int(np.count_nonzero(instance < 0))
        raw_instance_ids.update(int(value) for value in np.unique(instance))
        valid = (semantic >= 0) & (instance >= 0)
        valid_point_count += int(np.count_nonzero(valid))
        if not np.any(valid):
            continue
        sem_valid = semantic[valid]
        inst_valid = instance[valid]
        unknown = np.setdiff1d(np.unique(sem_valid), known_ids, assume_unique=False)
        if unknown.size:
            raise ValueError(f"Data contains semantic IDs missing from label_map.json in {path}: {unknown.tolist()}")
        for semantic_id, count in zip(*np.unique(sem_valid, return_counts=True)):
            semantic_point_counts[int(semantic_id)] += int(count)
        unique_instances, instance_counts = np.unique(inst_valid, return_counts=True)
        for instance_id, count in zip(unique_instances, instance_counts):
            instance_id_int = int(instance_id)
            counts_by_instance[instance_id_int] = counts_by_instance.get(instance_id_int, 0) + int(count)
        base = max_semantic_id + 1
        pair_codes, pair_counts = np.unique(inst_valid * base + sem_valid, return_counts=True)
        for code, count in zip(pair_codes, pair_counts):
            instance_id_int = int(code // base)
            semantic_id_int = int(code % base)
            histogram = semantic_histograms.setdefault(
                instance_id_int, np.zeros(max_semantic_id + 1, dtype=np.int64)
            )
            histogram[semantic_id_int] += int(count)

    minimum_points = int(config["filtering"]["minimum_instance_points"])
    retained_ids = np.asarray(
        sorted(instance_id for instance_id, count in counts_by_instance.items() if count >= minimum_points),
        dtype=np.int64,
    )
    row_by_id = {int(instance_id): row for row, instance_id in enumerate(retained_ids)}
    row_count = len(retained_ids)
    sums = np.zeros((row_count, 3), dtype=np.float64)
    minima = np.full((row_count, 3), np.inf, dtype=np.float64)
    maxima = np.full((row_count, 3), -np.inf, dtype=np.float64)
    max_samples = int(sampling["max_points_per_instance"])
    strides = np.asarray(
        [max(1, int(np.ceil(counts_by_instance[int(instance_id)] / max_samples))) for instance_id in retained_ids],
        dtype=np.int64,
    )
    seen = np.zeros(row_count, dtype=np.int64)
    samples: list[list[np.ndarray]] = [[] for _ in range(row_count)]
    sampled_counts = np.zeros(row_count, dtype=np.int64)

    for _, xyz, semantic, instance in reader.iter_points():
        valid = (semantic >= 0) & (instance >= 0)
        if not np.any(valid) or row_count == 0:
            continue
        xyz_valid = xyz[valid]
        inst_valid = instance[valid]
        candidate = np.isin(inst_valid, retained_ids, assume_unique=False)
        xyz_valid = xyz_valid[candidate]
        inst_valid = inst_valid[candidate]
        if not len(inst_valid):
            continue
        rows = np.searchsorted(retained_ids, inst_valid)
        np.add.at(sums, rows, xyz_valid)
        np.minimum.at(minima, rows, xyz_valid)
        np.maximum.at(maxima, rows, xyz_valid)
        order = np.argsort(rows, kind="stable")
        sorted_rows = rows[order]
        sorted_xyz = xyz_valid[order]
        boundaries = np.flatnonzero(np.diff(sorted_rows)) + 1
        starts = np.r_[0, boundaries]
        stops = np.r_[boundaries, sorted_rows.size]
        for start, stop in zip(starts, stops):
            row = int(sorted_rows[start])
            local = sorted_xyz[start:stop]
            offset = int(seen[row])
            stride = int(strides[row])
            indices = np.arange(local.shape[0], dtype=np.int64)
            choose = ((offset + indices) % stride) == 0
            remaining = max_samples - int(sampled_counts[row])
            if remaining > 0 and np.any(choose):
                selected = local[choose][:remaining]
                samples[row].append(selected)
                sampled_counts[row] += selected.shape[0]
            seen[row] += local.shape[0]

    records: list[InstanceRecord] = []
    semantic_instance_counts = {int(key): 0 for key in known_ids}
    tie_count = 0
    for row, instance_id in enumerate(retained_ids):
        instance_id_int = int(instance_id)
        histogram = semantic_histograms[instance_id_int]
        maximum = int(histogram.max())
        tied_ids = np.flatnonzero(histogram == maximum)
        semantic_id = int(tied_ids.min())
        semantic_tie = tied_ids.size > 1
        tie_count += int(semantic_tie)
        point_count = counts_by_instance[instance_id_int]
        semantic_instance_counts[semantic_id] += 1
        sample_points = np.concatenate(samples[row], axis=0) if samples[row] else np.empty((0, 3), dtype=np.float64)
        records.append(InstanceRecord(
            instance_id=instance_id_int,
            semantic_id=semantic_id,
            semantic_class=label_map.semantic_id_to_name[semantic_id],
            semantic_purity=float(maximum / point_count),
            point_count=point_count,
            centroid=sums[row] / point_count,
            xyz_min=minima[row],
            xyz_max=maxima[row],
            sample_points=sample_points,
            semantic_tie=semantic_tie,
        ))

    stats = {
        "raw_point_count": reader.schema.point_count,
        "semantic_minus1_point_count": semantic_minus1_count,
        "negative_instance_point_count": negative_instance_count,
        "valid_point_count": valid_point_count,
        "raw_unique_instance_count": len(raw_instance_ids),
        "valid_instance_count": len(records),
        "semantic_tie_instance_count": tie_count,
        "below_minimum_points_instance_count": len(counts_by_instance) - len(records),
    }
    return BuildingInstances(
        source_path=Path(path).resolve(),
        records=records,
        stats=stats,
        semantic_point_counts={key: value for key, value in semantic_point_counts.items() if value},
        semantic_instance_counts={key: value for key, value in semantic_instance_counts.items() if value},
        schema=reader.schema,
    )


@dataclass
class AssignmentResult:
    building: BuildingInstances
    local_frame: Any
    geometry: dict[int, Any]
    graph: Any
    applicable_mask: np.ndarray
    semantic_prior_score: np.ndarray
    geometry_score: np.ndarray
    context_score: np.ndarray
    confidence: np.ndarray
    vector: np.ndarray
    conditions: np.ndarray
    sources: np.ndarray


def _write_scalar_string(group: h5py.Group, name: str, value: str) -> None:
    group.create_dataset(name, data=value, dtype=h5py.string_dtype(encoding="utf-8"))


def write_functional_h5(
    path: str | Path,
    result: AssignmentResult,
    source_relative_path: str,
    original_ontology: Ontology,
    used_ontology: Ontology,
    label_map_path: str | Path,
    label_map: LabelMap,
    config: dict[str, Any],
    config_path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    source = result.building.source_path
    string_dtype = h5py.string_dtype(encoding="utf-8")
    compression = config["output"]["compression"]
    compression_level = int(config["output"]["compression_level"])
    with h5py.File(target, "w", libver="earliest") as handle:
        metadata = handle.create_group("metadata", track_order=True)
        _write_scalar_string(metadata, "source_building_filename", source.name)
        _write_scalar_string(metadata, "source_building_relative_path", source_relative_path)
        _write_scalar_string(metadata, "source_building_sha256", sha256_file(source))
        _write_scalar_string(metadata, "original_ontology_filename", original_ontology.path.name)
        _write_scalar_string(metadata, "original_ontology_sha256", original_ontology.sha256)
        _write_scalar_string(metadata, "ontology_filename", used_ontology.path.name)
        _write_scalar_string(metadata, "ontology_sha256", used_ontology.sha256)
        label_map_path = Path(label_map_path).resolve()
        _write_scalar_string(metadata, "label_map_filename", label_map_path.name)
        _write_scalar_string(metadata, "label_map_sha256", sha256_file(label_map_path))
        config_path = Path(config_path).resolve()
        _write_scalar_string(metadata, "config_filename", config_path.name)
        _write_scalar_string(metadata, "config_sha256", sha256_file(config_path))
        metadata.create_dataset("attribute_names", data=np.asarray(used_ontology.attribute_names, dtype=object), dtype=string_dtype)
        metadata.create_dataset("attribute_count", data=np.int32(len(used_ontology.attribute_names)))
        metadata.create_dataset("semantic_class_names", data=np.asarray(used_ontology.semantic_classes, dtype=object), dtype=string_dtype)
        _write_scalar_string(metadata, "creation_time", str(config["output"]["deterministic_creation_time"]))
        _write_scalar_string(metadata, "software_version", __version__)
        _write_scalar_string(metadata, "zero_vector_meaning",
                             "auxiliary binary attribute_value (confidence >= 0.5); a 0 is an automatic suggestion "
                             "below 0.5 and does NOT mean unassigned - every applicable attribute is computed by the "
                             "automatic rules")
        _write_scalar_string(metadata, "fusion", "odds_fusion: conf = P*G*C / (P*G*C + (1-P)*(1-G)*(1-C))")
        _write_scalar_string(metadata, "vector_rule", "vector = confidence >= 0.5 (auxiliary; continuous confidence is the primary result)")

        instances = handle.create_group("instances", track_order=True)
        records = result.building.records
        kwargs = {"compression": compression, "compression_opts": compression_level, "shuffle": True}
        instances.create_dataset("instance_id", data=np.asarray([r.instance_id for r in records], dtype=np.int64), **kwargs)
        instances.create_dataset("semantic_id", data=np.asarray([r.semantic_id for r in records], dtype=np.int32), **kwargs)
        instances.create_dataset("semantic_class", data=np.asarray([r.semantic_class for r in records], dtype=object), dtype=string_dtype, **kwargs)
        instances.create_dataset("semantic_purity", data=np.asarray([r.semantic_purity for r in records], dtype=np.float32), **kwargs)
        instances.create_dataset("point_count", data=np.asarray([r.point_count for r in records], dtype=np.int64), **kwargs)
        instances.create_dataset("applicable_mask", data=result.applicable_mask.astype(np.uint8), **kwargs)
        instances.create_dataset("semantic_prior_score", data=result.semantic_prior_score.astype(np.float32), **kwargs)
        instances.create_dataset("geometry_score", data=result.geometry_score.astype(np.float32), **kwargs)
        instances.create_dataset("context_score", data=result.context_score.astype(np.float32), **kwargs)
        instances.create_dataset("functional_attribute_confidence", data=result.confidence.astype(np.float32), **kwargs)
        instances.create_dataset("functional_attribute_vector", data=result.vector.astype(np.uint8), **kwargs)
    return target


CSV_FIELDS = (
    "building_id", "instance_id", "semantic_id", "semantic_class", "semantic_purity", "point_count",
    "attribute_name", "applicable", "semantic_prior_score", "geometry_score", "context_score",
    "final_confidence", "attribute_value", "condition", "source",
)


def write_functional_csv(path: str | Path, result: AssignmentResult, ontology: Ontology) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    building_id = result.building.source_path.stem
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row, record in enumerate(result.building.records):
            for column, attribute in enumerate(ontology.attribute_names):
                writer.writerow({
                    "building_id": building_id,
                    "instance_id": record.instance_id,
                    "semantic_id": record.semantic_id,
                    "semantic_class": record.semantic_class,
                    "semantic_purity": f"{record.semantic_purity:.8f}",
                    "point_count": record.point_count,
                    "attribute_name": attribute,
                    "applicable": int(result.applicable_mask[row, column]),
                    "semantic_prior_score": f"{result.semantic_prior_score[row, column]:.8f}",
                    "geometry_score": f"{result.geometry_score[row, column]:.8f}",
                    "context_score": f"{result.context_score[row, column]:.8f}",
                    "final_confidence": f"{result.confidence[row, column]:.8f}",
                    "attribute_value": int(result.vector[row, column]),
                    "condition": result.conditions[row, column],
                    "source": result.sources[row, column],
                })
    return target


def validate_score_arrays(*arrays: np.ndarray) -> None:
    for array in arrays:
        if not np.all(np.isfinite(array)):
            raise ValueError("Score array contains NaN or infinity")
        if np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError("Score array is outside [0,1]")


REQUIRED_INSTANCE_DATASETS = {
    "instance_id", "semantic_id", "semantic_class", "semantic_purity", "point_count", "applicable_mask",
    "semantic_prior_score", "geometry_score", "context_score", "functional_attribute_confidence",
    "functional_attribute_vector",
}


def _decode_array(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def validate_output_file(
    path: str | Path,
    label_map: LabelMap,
    ontology: Ontology,
    config: dict[str, Any],
    config_path: str | Path | None = None,
    label_map_path: str | Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    output_path = Path(path)
    with h5py.File(output_path, "r") as handle:
        if set(handle) != {"metadata", "instances"}:
            errors.append(f"unexpected root groups: {sorted(handle)}")
        instances = handle["instances"]
        if set(instances) != REQUIRED_INSTANCE_DATASETS:
            errors.append(f"instance dataset mismatch: {sorted(set(instances) ^ REQUIRED_INSTANCE_DATASETS)}")
        if int(handle["metadata/attribute_count"][()]) != len(CANONICAL_ATTRIBUTE_ORDER):
            errors.append("metadata attribute_count is not the fixed value 15")
        ids = instances["instance_id"][:]
        semantic_ids = instances["semantic_id"][:]
        semantic_classes = _decode_array(instances["semantic_class"][:])
        purity = instances["semantic_purity"][:]
        applicable = instances["applicable_mask"][:]
        prior = instances["semantic_prior_score"][:]
        geometry = instances["geometry_score"][:]
        context = instances["context_score"][:]
        confidence = instances["functional_attribute_confidence"][:]
        vector = instances["functional_attribute_vector"][:]
        expected_shape = (len(ids), len(ontology.attribute_names))
        for name, array in {
            "applicable_mask": applicable, "semantic_prior_score": prior, "geometry_score": geometry,
            "context_score": context, "functional_attribute_confidence": confidence,
            "functional_attribute_vector": vector,
        }.items():
            if array.shape != expected_shape:
                errors.append(f"{name} shape {array.shape} != {expected_shape}")
        if np.any(ids < 0):
            errors.append("negative output instance_id")
        if len(np.unique(ids)) != len(ids):
            errors.append("duplicate output instance_id")
        if len(ids) > 1 and np.any(np.diff(ids) <= 0):
            errors.append("output instance_id is not strictly increasing")
        if np.any(semantic_ids < 0):
            errors.append("negative output semantic_id")
        for semantic_id, semantic_class in zip(semantic_ids, semantic_classes):
            expected = label_map.semantic_id_to_name.get(int(semantic_id))
            if expected != semantic_class:
                errors.append(f"semantic mapping mismatch: {semantic_id} -> {semantic_class!r}, expected {expected!r}")
                break
        if np.any(~np.isfinite(purity)) or np.any((purity < 0) | (purity > 1)):
            errors.append("invalid semantic_purity")
        for name, array in {"prior": prior, "geometry": geometry, "context": context, "confidence": confidence}.items():
            if np.any(~np.isfinite(array)) or np.any((array < 0) | (array > 1)):
                errors.append(f"invalid {name} values")
        if np.any((applicable != 0) & (applicable != 1)) or np.any((vector != 0) & (vector != 1)):
            errors.append("mask/vector is not binary")
        # Auxiliary vector: confidence >= 0.5; not-applicable forced to 0.
        expected_vector = np.zeros_like(vector)
        expected_vector[...] = (confidence >= 0.5).astype(np.uint8)
        expected_vector[applicable == 0] = 0
        if not np.array_equal(vector, expected_vector):
            errors.append("vector is inconsistent with confidence >= 0.5")
        non_applicable = applicable == 0
        if any(np.any(array[non_applicable]) for array in (prior, geometry, context, confidence)):
            errors.append("non-applicable attribute has non-zero P/G/C/confidence")

        if all(array.shape == expected_shape for array in
               (applicable, prior, geometry, context, confidence)):
            expected_applicable = np.zeros_like(applicable)
            expected_prior = np.zeros_like(prior, dtype=np.float64)
            always_mask = np.zeros_like(applicable, dtype=bool)
            for row, semantic_class in enumerate(semantic_classes):
                by_attribute = ontology.rules_by_class_attribute.get(semantic_class, {})
                for attribute, rules in by_attribute.items():
                    column = ontology.attribute_index[attribute]
                    expected_applicable[row, column] = 1
                    always_mask[row, column] = all(rule.condition == "always" for rule in rules)
                    if always_mask[row, column]:
                        base = float(config["semantic_priors"]["by_class"][semantic_class][attribute])
                        expected_prior[row, column] = 0.5 + float(purity[row]) * (base - 0.5)
                    else:
                        expected_prior[row, column] = 0.5
            if not np.array_equal(applicable, expected_applicable):
                errors.append("applicable_mask does not match ontology class-attribute rules")
            active = expected_applicable == 1
            if np.any(np.abs(prior[active] - expected_prior[active]) > 2e-6):
                errors.append("semantic_prior_score does not match stable semantic prior / neutral refinement prior")

            from scoring import odds_fusion
            recomputed = np.zeros_like(confidence, dtype=np.float64)
            epsilon = float(config["numerical"]["odds_epsilon"])
            for row, column in zip(*np.nonzero(active)):
                recomputed[row, column] = odds_fusion(
                    float(prior[row, column]), float(geometry[row, column]),
                    float(context[row, column]), epsilon,
                )
            if np.any(np.abs(confidence[active] - recomputed[active]) > 5e-6):
                errors.append("functional confidence is inconsistent with odds fusion of stored P/G/C")
        if (np.any(np.abs(geometry[always_mask] - 0.5) > 1e-7)
                or np.any(np.abs(context[always_mask] - 0.5) > 1e-7)
                or np.any(np.abs(confidence[always_mask] - prior[always_mask]) > 2e-6)):
            errors.append("always attribute does not preserve semantic prior with G=C=0.5")

        other_rows = np.asarray([name == "other" for name in semantic_classes], dtype=bool)
        if np.any(applicable[other_rows]) or np.any(confidence[other_rows]) or np.any(vector[other_rows]):
            errors.append("other class is not all-zero")
        if "point_cloud" in handle or any("xyz" in name.lower() for name in handle):
            errors.append("output contains point-cloud data")
        def metadata_text(name: str) -> str:
            value = handle[f"metadata/{name}"][()]
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)
        if metadata_text("ontology_sha256") != ontology.sha256:
            errors.append("metadata ontology_sha256 does not match current ontology")
        if config_path is not None:
            if metadata_text("config_sha256") != sha256_file(config_path):
                errors.append("metadata config_sha256 does not match current config")
        if label_map_path is not None:
            if metadata_text("label_map_sha256") != sha256_file(label_map_path):
                errors.append("metadata label_map_sha256 does not match current label map")
        metadata_attributes = _decode_array(handle["metadata/attribute_names"][:])
        if metadata_attributes != list(ontology.attribute_names):
            errors.append("metadata attribute order does not match used ontology")
    return {
        "path": str(output_path),
        "valid": not errors,
        "errors": errors,
        "instance_count": int(len(ids)),
        "attribute_count": len(ontology.attribute_names),
    }


def validate_csv_against_h5(csv_path: str | Path, h5_path: str | Path, ontology: Ontology) -> dict[str, Any]:
    errors: list[str] = []
    with h5py.File(h5_path, "r") as handle:
        instance_ids = handle["instances/instance_id"][:]
        confidence = handle["instances/functional_attribute_confidence"][:]
        vector = handle["instances/functional_attribute_vector"][:]
        applicable = handle["instances/applicable_mask"][:]
    expected_rows = len(instance_ids) * len(ontology.attribute_names)
    count = 0
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for count, row in enumerate(reader, start=1):
            zero_based = count - 1
            instance_row = zero_based // len(ontology.attribute_names)
            column = zero_based % len(ontology.attribute_names)
            if instance_row >= len(instance_ids):
                errors.append("CSV has extra rows")
                break
            if int(row["instance_id"]) != int(instance_ids[instance_row]):
                errors.append(f"CSV instance mismatch at row {count}")
                break
            if row["attribute_name"] != ontology.attribute_names[column]:
                errors.append(f"CSV attribute order mismatch at row {count}")
                break
            if int(row["applicable"]) != int(applicable[instance_row, column]):
                errors.append(f"CSV applicable mismatch at row {count}")
                break
            if abs(float(row["final_confidence"]) - float(confidence[instance_row, column])) > 1e-6:
                errors.append(f"CSV confidence mismatch at row {count}")
                break
            if int(row["attribute_value"]) != int(vector[instance_row, column]):
                errors.append(f"CSV vector mismatch at row {count}")
                break
    if count != expected_rows:
        errors.append(f"CSV row count {count} != {expected_rows}")
    return {"valid": not errors, "errors": errors, "row_count": count, "expected_row_count": expected_rows}
