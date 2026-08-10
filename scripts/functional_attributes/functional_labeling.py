#!/usr/bin/env python3
"""HFX3D functional attribute refinement - main entry point.

Runs the non-learning refinement pipeline over building point clouds:
semantic prior + geometry evidence + graph-context evidence are fused with a
parameter-free odds fusion into a 15-dimensional functional-attribute
confidence vector per instance. Results are written as H5 and CSV files.

Run from the repository root:

    python scripts/functional_attributes/functional_labeling.py \\
        --data-root /path/to/HFX3D \\
        --output-root /path/to/output
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow `python scripts/functional_attributes/functional_labeling.py` from any CWD.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from hfx3d_core import (
    LabelMap,
    Ontology,
    __version__,
    load_config,
    load_label_map,
    load_ontology,
    sha256_file,
    validate_csv_against_h5,
    validate_output_file,
    write_functional_csv,
    write_functional_h5,
)
from hfx3d_rules import assign_building

# ============================================================
# Default paths (resolved from repository root)
# ============================================================
# scripts/functional_attributes/ -> parents[2] = repository root
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ONTOLOGY_PATH = REPO_ROOT / "configs" / "functional_attributes" / "ontology_rules.yaml"
DEFAULT_LABEL_MAP_PATH = REPO_ROOT / "configs" / "label_map.json"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "functional_attributes" / "default_config.yaml"


def find_selected_buildings(data_root: Path, selected: list[str] | None) -> list[Path]:
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root}")
    all_files = sorted(
        p for p in data_root.rglob("*") if p.suffix.lower() in {".h5", ".hdf5"}
    )
    if not all_files:
        raise FileNotFoundError(f"No H5/HDF5 files under {data_root}")
    if selected is None:
        return all_files
    by_stem: dict[str, Path] = {}
    for path in all_files:
        by_stem[path.stem] = path
        by_stem[path.name] = path
    chosen: list[Path] = []
    for item in selected:
        name = item.strip()
        if not name:
            continue
        stem = Path(name).stem if name.lower().endswith((".h5", ".hdf5")) else name
        key = name if name in by_stem else stem
        if key not in by_stem and stem not in by_stem:
            raise FileNotFoundError(f"Building not found: {name}")
        path = by_stem.get(key) or by_stem[stem]
        if path not in chosen:
            chosen.append(path)
    if not chosen:
        raise ValueError("No buildings selected after parsing")
    return chosen


def prepare_output_directories(output_root: Path) -> tuple[Path, Path]:
    output_root = Path(output_root)
    h5_root = output_root / "functional_labels"
    csv_root = output_root / "functional_labels_csv"
    h5_root.mkdir(parents=True, exist_ok=True)
    csv_root.mkdir(parents=True, exist_ok=True)
    return h5_root, csv_root


def process_one_building(
    source_path: Path,
    data_root: Path,
    h5_root: Path,
    csv_root: Path,
    label_map: LabelMap,
    ontology: Ontology,
    original_ontology: Ontology,
    config: dict[str, Any],
    config_path: Path,
    label_map_path: Path,
    overwrite: bool,
) -> dict[str, Any] | None:
    source_path = Path(source_path)
    relative = source_path.relative_to(data_root)
    split = source_path.parent.name if source_path.parent != data_root else "all"
    out_h5 = h5_root / split / f"{source_path.stem}.h5"
    out_csv = csv_root / split / f"{source_path.stem}.csv"

    if not overwrite and (out_h5.exists() or out_csv.exists()):
        print(f"Skipped existing output: {out_h5}")
        return None

    result = assign_building(source_path, label_map, ontology, config)

    write_functional_h5(
        out_h5,
        result,
        str(relative),
        original_ontology,
        ontology,
        label_map_path,
        label_map,
        config,
        config_path,
    )
    write_functional_csv(out_csv, result, ontology)

    summary = {
        "building": source_path.stem,
        "split": split,
        "source_relative": str(relative),
        "instance_count": len(result.building.records),
        "output_h5": str(out_h5),
        "output_csv": str(out_csv),
    }
    print(
        f"built {source_path.stem}: {summary['instance_count']} instances -> {out_h5}",
        flush=True,
    )
    return summary


def build_functional_labels(
    source_files: list[Path],
    output_root: Path,
    ontology: Ontology,
    label_map: LabelMap,
    config: dict[str, Any],
    data_root: Path,
    config_path: Path,
    label_map_path: Path,
    ontology_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    h5_root, csv_root = prepare_output_directories(output_root)
    original = load_ontology(ontology_path)

    manifest_files: list[dict[str, Any]] = []
    total_instances = 0
    for source in source_files:
        item = process_one_building(
            source_path=source,
            data_root=data_root,
            h5_root=h5_root,
            csv_root=csv_root,
            label_map=label_map,
            ontology=ontology,
            original_ontology=original,
            config=config,
            config_path=config_path,
            label_map_path=label_map_path,
            overwrite=overwrite,
        )
        if item is None:
            continue
        total_instances += int(item["instance_count"])
        manifest_files.append(item)

    summary = {
        "building_count": len(manifest_files),
        "instance_count": total_instances,
        "attribute_names": list(ontology.attribute_names),
        "attribute_count": len(ontology.attribute_names),
        "ontology_sha256": ontology.sha256,
        "config_sha256": sha256_file(config_path),
        "package_version": __version__,
        "output_root": str(output_root),
        "files": manifest_files,
    }
    print(json.dumps(
        {"status": "ok", "buildings": summary["building_count"], "instances": total_instances},
        indent=2,
    ))
    return summary


def validate_one_building(
    h5_path: Path,
    csv_path: Path,
    label_map: LabelMap,
    ontology: Ontology,
    config: dict[str, Any],
    config_path: Path,
    label_map_path: Path,
) -> dict[str, Any]:
    report = validate_output_file(h5_path, label_map, ontology, config, config_path, label_map_path)
    errors = list(report.get("errors") or [])
    if csv_path.exists():
        csv_report = validate_csv_against_h5(csv_path, h5_path, ontology)
        if not csv_report["valid"]:
            errors.extend(csv_report.get("errors") or [])
    else:
        errors.append("missing csv")
    return {
        "file": str(h5_path),
        "csv": str(csv_path),
        "valid": not errors,
        "errors": errors,
        "instance_count": int(report.get("instance_count", 0)),
    }


def validate_generated_results(
    source_files: list[Path],
    output_root: Path,
    ontology: Ontology,
    label_map: LabelMap,
    config: dict[str, Any],
    config_path: Path,
    label_map_path: Path,
) -> dict[str, Any]:
    h5_root = output_root / "functional_labels"
    csv_root = output_root / "functional_labels_csv"
    if not h5_root.is_dir():
        raise FileNotFoundError(f"No functional_labels under {output_root}")

    selected_stems = {p.stem for p in source_files}
    h5_files = sorted(h5_root.rglob("*.h5"))
    if selected_stems:
        h5_files = [p for p in h5_files if p.stem in selected_stems] or h5_files

    file_reports: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    total = 0
    for h5_path in h5_files:
        csv_path = csv_root / h5_path.relative_to(h5_root).with_suffix(".csv")
        item = validate_one_building(h5_path, csv_path, label_map, ontology, config,
                                     config_path, label_map_path)
        total += int(item["instance_count"])
        file_reports.append(item)
        if not item["valid"]:
            errors.append({"file": item["file"], "errors": item["errors"]})

    out = {
        "valid": not errors,
        "h5_count": len(h5_files),
        "instance_count": total,
        "attribute_count": len(ontology.attribute_names),
        "errors": errors,
        "files": file_reports,
    }
    print(json.dumps(
        {k: out[k] for k in ("valid", "h5_count", "instance_count", "attribute_count", "errors")},
        indent=2,
    ))
    return out


def run(
    data_root: Path,
    output_root: Path,
    ontology_path: Path = DEFAULT_ONTOLOGY_PATH,
    label_map_path: Path = DEFAULT_LABEL_MAP_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
    buildings: list[str] | None = None,
    do_build: bool = True,
    do_validate: bool = True,
    overwrite: bool = True,
) -> None:
    """Execute the pipeline using the given paths and flags."""
    data_root = Path(data_root)
    output_root = Path(output_root)
    ontology_path = Path(ontology_path)
    label_map_path = Path(label_map_path)
    config_path = Path(config_path)

    print("=== HFX3D functional attribute refinement ===")
    print(f"REPO_ROOT     = {REPO_ROOT}")
    print(f"DATA_ROOT     = {data_root}")
    print(f"OUTPUT_ROOT   = {output_root}")
    print(f"ONTOLOGY_PATH = {ontology_path}")
    print(f"LABEL_MAP     = {label_map_path}")
    print(f"CONFIG        = {config_path}")
    print(f"BUILDINGS     = {buildings if buildings is not None else '(all)'}")
    print(f"DO_BUILD      = {do_build}")
    print(f"DO_VALIDATE   = {do_validate}")
    print(f"OVERWRITE     = {overwrite}")

    for path, label in (
        (config_path, "config"),
        (label_map_path, "label map"),
        (ontology_path, "ontology"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    config = load_config(config_path)
    label_map = load_label_map(label_map_path)
    ontology = load_ontology(ontology_path)

    source_files = find_selected_buildings(data_root, buildings)
    print(f"Selected {len(source_files)} building(s): {[p.stem for p in source_files]}")

    if do_build:
        build_functional_labels(
            source_files=source_files,
            output_root=output_root,
            ontology=ontology,
            label_map=label_map,
            config=config,
            data_root=data_root,
            config_path=config_path,
            label_map_path=label_map_path,
            ontology_path=ontology_path,
            overwrite=overwrite,
        )
    if do_validate:
        validate_generated_results(
            source_files=source_files,
            output_root=output_root,
            ontology=ontology,
            label_map=label_map,
            config=config,
            config_path=config_path,
            label_map_path=label_map_path,
        )
    print("Finished.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "HFX3D functional attribute refinement. "
            "Provide --data-root and --output-root; config/ontology/label-map "
            "default to repository configs/."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory of HFX3D building HDF5 files (e.g. train/val/test layout)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory for functional_labels/ and functional_labels_csv/ outputs",
    )
    parser.add_argument(
        "--buildings",
        nargs="*",
        default=None,
        help="Optional building stems; omit to process all buildings under data-root",
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=DEFAULT_ONTOLOGY_PATH,
        help="Ontology YAML (default: configs/functional_attributes/ontology_rules.yaml)",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=DEFAULT_LABEL_MAP_PATH,
        help="Semantic label map JSON (default: configs/label_map.json)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Pipeline config YAML (default: configs/functional_attributes/default_config.yaml)",
    )
    parser.add_argument("--no-build", action="store_true", help="Skip writing outputs")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip buildings that already have H5/CSV outputs",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(
        data_root=args.data_root,
        output_root=args.output_root,
        ontology_path=args.ontology,
        label_map_path=args.label_map,
        config_path=args.config,
        buildings=args.buildings,
        do_build=not args.no_build,
        do_validate=not args.no_validate,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
