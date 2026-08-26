# HFX3D Functional Attribute Refinement

Automatic functional-attribute annotation / refinement for semantic building
instances in HFX3D point clouds.

## 1. Overview

This module assigns a **15-dimensional functional confidence vector** to each
semantic instance.

```text
semantic instance
→ semantic functional prior
→ geometry evidence
→ graph / context evidence
→ odds-domain fusion
→ 15-D functional confidence
→ automatic functional attribute suggestion
```

Outputs are instance-level HDF5 and CSV files.
## 2. Method

For each instance \(I\) and each applicable functional attribute \(a\):

- **P** — configured semantic prior for each semantic class–attribute pair
- **G** — instance-level geometric evidence in \([0, 1]\)
- **C** — instance-level contextual evidence in \([0, 1]\)
- **f** — confidence obtained through parameter-free odds fusion:

```text
f = P G C / [P G C + (1-P)(1-G)(1-C)]
```

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
├── functional_labeling.py             # functional attribute assignment entry point
├── hfx3d_core.py                      # H5/CSV I/O, ontology, and validation
├── hfx3d_rules.py                     # geometry and context refinement rules
├── scoring.py                         # continuous scoring and odds fusion
├── compute_ambiguity.py               # Train-only class-attribute ambiguity analysis
├── build_ablation_scores.py           # CSV/H5 loading and ablation score construction
├── evaluate_ablation.py               # per-attribute F1/AP and macro metric utilities
└── evaluate_functional_ablation.py    # Val/Test ablation evaluation entry point

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

### Ambiguity analysis

Define ambiguous class-attribute pairs from the human-validated training split:

```bash
python scripts/functional_attributes/compute_ambiguity.py \
    --validated-root /path/to/validated_h5 \
    --output-root /path/to/ambiguity_outputs \
    --tau 0.05
```

The script uses `eval_include=true` instances and applicable attribute cells.
It writes attribute rates, ambiguous pairs, ambiguous classes, and an audit
report to the output directory.

### Ablation evaluation

Evaluate the six evidence settings (`S only`, `G only`, `C only`, `S+G`, `S+C`,
and `S+G+C`) against the human-validated labels:

```bash
python scripts/functional_attributes/evaluate_functional_ablation.py \
    --prediction-csv-root /path/to/functional_labels_csv \
    --validated-h5-root /path/to/validated_h5 \
    --ambiguous-classes /path/to/ambiguity_outputs/ambiguous_classes.json \
    --output-root /path/to/evaluation_outputs
```

The evaluator reports mF1, mAP, and ESM for the Val and Test splits, including
the All and Ambiguous subsets. It also writes per-attribute F1 and AP results.
`build_ablation_scores.py` and `evaluate_ablation.py` provide the shared loading,
score construction, and metric functions used by this entry point.
