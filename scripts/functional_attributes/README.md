# HFX3D Functional Attribute Refinement

Automatic functional-attribute annotation / refinement for semantic building
instances in HFX3D point clouds.

## 1. Overview

This module assigns a **15-dimensional functional confidence vector** to each
semantic instance. The pipeline is **deterministic and non-learning**.

```text
semantic instance
→ semantic functional prior
→ geometry evidence
→ graph / context evidence
→ odds-domain fusion
→ 15-D functional confidence
→ automatic functional attribute suggestion
```

Outputs are instance-level HDF5 and CSV files. Original point coordinates,
semantic labels, and instance IDs are not modified.

## 2. Method

For each instance \(I\) and each applicable functional attribute \(a\):

- **P** — semantic prior (ontology + purity; instance-dependent attributes use neutral \(P = 0.5\))
- **G** — geometric evidence in \([0, 1]\)
- **C** — graph / contextual evidence in \([0, 1]\)
- **f** — fused confidence via parameter-free odds fusion:

```text
f = P G C / [P G C + (1-P)(1-G)(1-C)]
```

Neutral evidence is \(0.5\). The binary automatic suggestion is derived with the
shared threshold \(0.5\) on the continuous confidence for applicable attributes.

## 3. Functional attributes

Fixed 15-dimensional order (column index in every output matrix):

| Index | Attribute |
|---:|---|
| 0 | `load_bearing` |
| 1 | `thermal_envelope` |
| 2 | `vegetation_support` |
| 3 | `operable` |
| 4 | `solar_shading` |
| 5 | `ventilation` |
| 6 | `natural_lighting` |
| 7 | `access` |
| 8 | `drainage` |
| 9 | `fall_protection` |
| 10 | `aesthetic` |
| 11 | `privacy_screening` |
| 12 | `circulation` |
| 13 | `illumination` |
| 14 | `surveillance` |

## 4. Repository layout

```text
scripts/functional_attributes/
├── functional_labeling.py   # entry point
├── hfx3d_core.py            # I/O, ontology, writers, validation
├── hfx3d_rules.py           # geometry / context refinement
└── scoring.py               # continuous scoring + odds fusion

configs/
├── label_map.json           # shared 12-class semantic map
└── functional_attributes/
    ├── default_config.yaml
    └── ontology_rules.yaml
```

## 5. Installation

From the repository root:

```bash
pip install -r requirements.txt
```

## 6. Usage

Run from the **repository root** (working directory may be anywhere; paths are
resolved relative to this file and the repo root):

```bash
python scripts/functional_attributes/functional_labeling.py \
    --data-root /path/to/HFX3D \
    --output-root /path/to/output
```

`--data-root` should contain HFX3D building HDF5 files (typically `train/`,
`val/`, `test/`). Outputs are written under:

```text
<path/to/output>/functional_labels/
<path/to/output>/functional_labels_csv/
```

Optional advanced flags:

| Flag | Purpose |
|------|---------|
| `--buildings STEM ...` | Process only listed buildings |
| `--config PATH` | Override pipeline config YAML |
| `--ontology PATH` | Override ontology YAML |
| `--label-map PATH` | Override semantic label map |
| `--no-overwrite` | Skip buildings that already have outputs |
| `--no-build` | Validation only |
| `--no-validate` | Skip post-write validation |

Defaults (when not overridden):

```text
configs/label_map.json
configs/functional_attributes/default_config.yaml
configs/functional_attributes/ontology_rules.yaml
```
