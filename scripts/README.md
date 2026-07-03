# Instance & Classification Annotation Pipeline

Tools for adding **instance IDs** to the HFX3D HDF5 dataset and for **reviewing /
correcting** both instance and classification labels in CloudCompare.

Two scripts do everything:

| Script | Direction | Purpose |
|--------|-----------|---------|
| `generate_instances.py` | HDF5 → (HDF5 + PLY) | Cluster points into objects and/or export an editable PLY for CloudCompare |
| `import_instances.py` | PLY → HDF5 | Pull your CloudCompare edits back into the dataset |

---

## 1. How it works (read this first)

- **The HDF5 file is the source of truth.** It holds the real point cloud, the
  class labels (`labels/semantic`), and the instance labels (`labels/instance`).
- **The PLY is a disposable view.** It's what you open and edit in CloudCompare.
  It is regenerated from the HDF5 anytime; deleting it loses nothing.
- **The point-index is the glue.** Every point carries its original HDF5 row
  number, stored as two fields `scalar_PointIdxHi`/`scalar_PointIdxLo` (split so
  CloudCompare's float32 scalar storage stays exact even past 16.7M points).
  CloudCompare can reorder, split, or delete points freely — the import maps
  every edit back by this index. **Never edit it.** If the index is ever missing
  or corrupted, the importer automatically falls back to matching by XYZ.

Each editable PLY carries three scalar fields:

| Field | Meaning | Edit it? |
|-------|---------|----------|
| `scalar_Instance` | object id (`0` = no instance, `1…` = objects) | ✅ yes — merge/split/relabel |
| `scalar_Classification` | class `0–11` (same as HDF5 `labels/semantic`) | ✅ yes — fix wrong classes |
| `scalar_PointIdxHi` / `scalar_PointIdxLo` | original HDF5 row number (split for float32 safety) | ❌ never |

### Instance IDs
- **Scene-local:** unique *within one building* only (building A's "5" and
  building B's "5" are unrelated). This is standard for instance segmentation.
- **`0` = no instance** (deleted points, or clusters below `--min-points`).
- **IDs are preserved on import** — if you merge a piece into object `35`, it
  stays `35`. Gaps are fine. Only `import --repack` renumbers to `1..K`, and you
  run that once at the very end if you want clean numbering.

### Classification = the class label
`scalar_Classification` (in the PLY) and `labels/semantic` (in the HDF5) are the
**same thing** — the per-point class, identical to your LAS `classification`.
The HDF5 keeps the name `labels/semantic` because the model loaders depend on it.

---

## 2. Setup

From the repo root, activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1      # PowerShell
# or, if activation is blocked, prefix every command with:
#   .venv\Scripts\python.exe  instead of  python
```

Dependencies (already in the venv): `numpy`, `scipy`, `scikit-learn`, `h5py`.
Colors come from `configs/palette.py` (the canonical 12-class palette).

The scripts read/write the dataset under `Output/` by default
(`Output/{train,val,test}/*.h5`). Editable PLYs are written to
`Output/instances_vis/{split}/<name>_instances_vis.ply`.

---

## 3. The workflow

```
 generate (draft)  →  view in CloudCompare  →  edit  →  import  →  re-view (--from-h5)
      once per building            repeat as needed
```

### Step 1 — Draft the instances (once per building)
Cluster a building into objects, write `labels/instance` into its HDF5, and
export the editable PLY:

```powershell
python scripts\generate_instances.py --only BLD003 --eps 0.20
```

Do **all** buildings at once (skip the big PLYs; make them per-building later):

```powershell
python scripts\generate_instances.py --eps 0.20 --no-vis
```

> ⚠️ Plain `generate` **re-clusters from scratch and overwrites** any hand-edits.
> Only run it for the *first* draft of a building. After editing, always use
> `--from-h5` (Step 5).

### Step 2 — Open in CloudCompare
Open `Output\instances_vis\train\HFX_BLD003_ZEB_CLEAN_instances_vis.ply`.
Each object has a distinct color (instance view).

### Step 3 — Edit
See the CloudCompare guide (Section 4 below). Edit `scalar_Instance` (and/or
`scalar_Classification`), then **File → Save → .ply**, same name.

### Step 4 — Import your edits
```powershell
# preview (writes nothing):
python scripts\import_instances.py "Output\instances_vis\train\HFX_BLD003_ZEB_CLEAN_instances_vis.ply" --root Output --dry-run
# commit:
python scripts\import_instances.py "Output\instances_vis\train\HFX_BLD003_ZEB_CLEAN_instances_vis.ply" --root Output
```
Add `--classification` to also pull class-label fixes back in (see Section 5).

### Step 5 — Re-view (preserves your edits)
```powershell
python scripts\generate_instances.py --only BLD003 --from-h5
```
`--from-h5` rebuilds the PLY *from the committed HDF5 labels* — it does **not**
re-cluster, so your edits survive. Reopen in CloudCompare to confirm.

---

## 4. Editing in CloudCompare

### Choosing what the colors mean
Regenerate with `--color` (with `--from-h5` to keep edits):

| `--color` | RGB shows | Use to spot |
|-----------|-----------|-------------|
| `instance` (default) | distinct color per object | merged / split objects |
| `class` | official 12-class palette | wrong class labels |
| `real` | true photo colors | physical context |

You can also leave the RGB as-is and **color by a scalar field** inside CC:
Properties → Scalar Fields → pick `scalar_Instance` or `scalar_Classification`.

### Relabeling instances
Edit the `scalar_Instance` value of the points you select:

| Action | Set `scalar_Instance` to |
|--------|--------------------------|
| **Merge** B into A | A's existing number (e.g. `35`) |
| **Split** off a new object | any unused number **above the current max** (e.g. `1001`, `1002`, …) |
| **Delete** junk | `0`, or just delete the points |

You never need a "correct" number — only the grouping matters. To see the
current max id:
```powershell
python -c "import h5py,numpy as np; i=h5py.File(r'Output\train\HFX_BLD003_ZEB_CLEAN.h5')['labels/instance'][:]; print('max id:',int(i.max()))"
```

### Fixing classes
With `--color class` you'll *see* misclassifications. Segment the wrong region
and set its `scalar_Classification` to the correct id:

`0 wall · 1 window · 2 door · 3 balcony · 4 vegetation · 5 stairs · 6 terrain · 7 roof · 8 blinds · 9 other · 10 column · 11 arch`

### Saving
**File → Save → Polygon File Format (*.ply)** and choose **binary**.
CloudCompare exports all scalar fields automatically.

---

## 5. Review 2 — instances + classification together

One PLY holds both `scalar_Instance` and `scalar_Classification`, so you fix both
in a single CloudCompare session:

```powershell
# 1. class-colored review PLY
python scripts\generate_instances.py --only BLD003 --from-h5 --color class
# 2. in CC: fix classes (edit scalar_Classification) AND instances
#    (switch coloring to scalar_Instance, edit it). Save as PLY (binary).
# 3. import BOTH:
python scripts\import_instances.py "Output\instances_vis\train\HFX_BLD003_ZEB_CLEAN_instances_vis.ply" --root Output --classification
```
The import prints `Instances: …` and `Classification: N point label(s) changed`.
Points you didn't touch keep their original class; deleted points become
instance `0` (their class is left unchanged).

### Review metrics (original vs reviewed)
The first time you import a class edit for a building, `import_instances.py`
**freezes the pre-review labels** into `labels/semantic_original`. That baseline
lets you measure the review the same way `utils/review_las_files.py` did — now
from the HDF5.

`import_instances.py --classification` already prints a live one-line summary
(accuracy / kappa / points changed). For the full report:

```powershell
# one building:
python scripts\review_metrics.py --root Output --only BLD003
# whole dataset (per-building + aggregate rollup):
python scripts\review_metrics.py --root Output
```

It reports, per building **and** as a dataset aggregate:
- overall accuracy (fraction unchanged) and Cohen's Kappa (agreement)
- per-class IoU for **changed classes only** (matches the old script)
- top reassignments (`original -> reviewed`, e.g. `wall -> window`)
- class distributions before/after and a full confusion matrix

A machine-readable summary is written to `Output/review_metrics.json`. Buildings
never class-reviewed (no baseline) are listed as skipped.

---

## 6. Team roles (annotators vs. owner)

- **Annotators** only need **CloudCompare** and the PLY for their building. They
  edit `scalar_Instance` / `scalar_Classification`, save as PLY, and send
  it back. They never touch Python or the HDF5.
- **Owner** runs `import_instances.py` to merge each returned PLY into the master
  dataset, and `generate_instances.py --from-h5` to produce/refresh PLYs.

---

## 7. Verifying the data

Instances + class distribution for one building:
```powershell
python -c "import h5py,numpy as np; h=h5py.File(r'Output\train\HFX_BLD003_ZEB_CLEAN.h5'); i=h['labels/instance'][:]; s=h['labels/semantic'][:]; print('instances:',len(np.unique(i[i>0])),'| max id:',int(i.max()),'| zero pts:',int((i==0).sum())); u,c=np.unique(s,return_counts=True); print('class counts:',dict(zip(u.tolist(),c.tolist())))"
```

A per-building summary is also written to `Output/instance_stats.json` after a
generate run.

---

## 8. Command reference

### `generate_instances.py`
| Option | Default | Meaning |
|--------|---------|---------|
| `--root` | `Output` | dataset root with `train/ val/ test/` |
| `--only <str>` | (all) | process only files whose name contains `<str>` |
| `--eps <m>` | `0.20` | connected-components radius, metres |
| `--octree-level <n>` | off | use adaptive radius `bbox/2**n` (old behavior; `8`) instead of `--eps` |
| `--min-points <n>` | `50` | drop clusters smaller than this |
| `--no-reassign` | off | leave dropped clusters as `0` instead of attaching to nearest same-class instance |
| `--color {instance,class,real}` | `instance` | RGB coloring of the PLY |
| `--vis-only` | off | write only the PLY; never modify the HDF5 (for tuning) |
| `--from-h5` | off | export PLY from stored labels; do **not** re-cluster (use after editing) |
| `--no-vis` | off | skip the PLY (e.g. bulk drafting) |
| `--voxel <m>` | `0.05` | downsample size used internally to speed up clustering |

### `import_instances.py`
| Option | Default | Meaning |
|--------|---------|---------|
| `ply` | (required) | the edited PLY |
| `--root` | `Output` | dataset root (the matching `.h5` is found by name) |
| `--h5 <path>` | (auto) | target `.h5` explicitly (overrides `--root` lookup) |
| `--classification` | off | also import `scalar_Classification` → `labels/semantic` |
| `--repack` | off | renumber instance ids to gap-free `1..K` (final freeze only) |
| `--dry-run` | off | report changes without writing |

### `review_metrics.py`
| Option | Default | Meaning |
|--------|---------|---------|
| `--root` | `Output` | dataset root |
| `--only <str>` | (all) | only files whose name contains `<str>` |
| `--out <path>` | `<root>/review_metrics.json` | JSON summary path |

### `repack_instances.py` — final freeze
Run once at the end, on buildings you've finished. It does two things directly
in the HDF5 (no PLY needed):
1. **Repacks** instance ids to a gap-free `1..K` (grouping unchanged, e.g. `9001
   -> 81`).
2. **Drops** `labels/semantic_original` (the review-2 baseline).

⚠️ Once the baseline is dropped, `review_metrics.py` can't be recomputed — so
**run `review_metrics.py` first** to save the report, and **`--dry-run` first**
to preview.

| Option | Default | Meaning |
|--------|---------|---------|
| `--root` | `Output` | dataset root |
| `--only <str>` | (all) | only files whose name contains `<str>` |
| `--dry-run` | off | preview what would change without writing |

```powershell
python scripts\repack_instances.py --root Output --only BLD001 --dry-run   # preview one
python scripts\repack_instances.py --root Output --only BLD001             # freeze one
python scripts\repack_instances.py --root Output                           # freeze all finished
```

---

## 9. Quick-start pipeline (step by step)

### Setup (once per terminal)
```powershell
.venv\Scripts\Activate.ps1
```

### Step 1 — Draft instances
```powershell
# One building:
python scripts\generate_instances.py --only BLD016 --eps 0.20
# All buildings at once (no PLYs; make those per-building when editing):
python scripts\generate_instances.py --eps 0.20 --no-vis
```
Writes `labels/instance` into the HDF5(s). ⚠️ Only for the **first draft** — never re-run plain `generate` on a building you've edited (it re-clusters and wipes edits).

### Step 2 — Get the editable PLY
If you drafted with `--no-vis`, make the PLY for the building you're about to edit:
```powershell
python scripts\generate_instances.py --only BLD016 --from-h5 --color class
```
Open `Output\instances_vis\train\HFX_BLD016_ZEB_CLEAN_instances_vis.ply` in CloudCompare.

### Step 3 — Edit in CloudCompare
- **Classification (review 2):** color by `scalar_Classification`; segment a wrong region, set it to the right class id.
- **Instances:** color by `scalar_Instance`; merge (target's id) / split (any unused number above current max) / delete (`0`).
- Never touch `scalar_PointIndex`. **File → Save → .ply (binary)**, same name.

### Step 4 — Import edits
```powershell
# preview (writes nothing):
python scripts\import_instances.py "Output\instances_vis\train\HFX_BLD016_ZEB_CLEAN_instances_vis.ply" --root Output --classification --dry-run
# commit:
python scripts\import_instances.py "Output\instances_vis\train\HFX_BLD016_ZEB_CLEAN_instances_vis.ply" --root Output --classification
```
Drop `--classification` if you only changed instances.

### Step 5 — Review metrics
```powershell
python scripts\review_metrics.py --root Output --only BLD016   # one building
python scripts\review_metrics.py --root Output                 # whole dataset
```

### Step 6 — Re-view (preserves edits)
```powershell
python scripts\generate_instances.py --only BLD016 --from-h5 --color instance
```

**Loop = Steps 2 → 6 per building.** The golden rule: after editing, only re-view with `--from-h5`; never plain `generate`.

### Step 7 — Final freeze (once all buildings are done)
Run metrics first (so the report is saved), then repack:
```powershell
python scripts\review_metrics.py --root Output                        # save final report
python scripts\repack_instances.py --root Output --only BLD001 --dry-run  # preview
python scripts\repack_instances.py --root Output --only BLD001            # freeze one
python scripts\repack_instances.py --root Output                          # freeze all
```
Repacks instance ids to gap-free `1..K` and drops `labels/semantic_original`. ⚠️ One-time — not part of the editing loop.

---

## 10. Rules of thumb (avoid the common mistakes)

1. **After any edit, re-view with `--from-h5`.** Plain `generate` re-clusters and
   wipes edits.
2. **Don't re-draft an already-annotated building** (same reason).
3. **`0` is reserved** for "no instance" — never assign it to a real object.
4. **Split → use a number above the current max** so it doesn't merge by accident.
5. **`--repack` is a one-time final step**, not part of the editing loop.
6. **Save as PLY** (binary) from CloudCompare and keep the scalar fields.
