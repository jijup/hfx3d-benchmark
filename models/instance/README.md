# Instance segmentation baselines

Learned instance-segmentation baselines for the HFX3D benchmark: **PointGroup**,
**ISBNet**, and **SoftGroup**. All three are trained end-to-end from raw point
features (no ground-truth semantics supplied) under one shared protocol and
scored with one shared evaluator.

See the paper for the full
protocol (5&nbsp;cm voxels, 7-D input `[x,y,z,r,g,b,intensity]`, 15&nbsp;m / 250k-point
crops, 18k-step budget, seed 42, checkpoint selected on validation, test set
touched once).

## Results (test set)

| Method | mAP | AP50 | AP25 |
|--------|:---:|:----:|:----:|
| SoftGroup  | **0.2032** | **0.2923** | **0.4044** |
| ISBNet     | 0.0422 | 0.0883 | 0.2276 |
| PointGroup | 0.0145 | 0.0427 | 0.0955 |

## How this is organised

Each method is a small set of **HFX3D-specific files plus a patch** applied on
top of its upstream repository — we do not vendor the full upstream trees. Every
method directory contains the HFX3D configs, the dataset loader, the full-scene
evaluation script, the data converter/preparation script (ISBNet/SoftGroup), and
the patch/diff capturing every change made against upstream. SLURM submission
scripts for the reported runs are under
[`slurm/instance/<method>/`](../../slurm/instance).

## Shared evaluator

All three methods score with
[`evaluation/insseg_eval_utils.py`](../../evaluation/insseg_eval_utils.py), a
ScanNet-v2 instance-AP port adapted for HFX3D: AP over IoU 0.50:0.05:0.90 plus
AP50 and AP25, with two HFX3D adaptations — void points are the stuff classes
**and** `instance == -1` (thing instances below the 200-point size floor), and
predictions are propagated to every point at full resolution (1-NN) **before**
scoring. Scoring sub-sampled predictions against full-resolution ground truth
otherwise caps mask IoU and collapses AP toward zero (the main bug behind the
near-zero pre-rerun numbers). Minimum region size is 100 points on the
prediction side; all ground-truth instances are ≥ 200.

Each method's `test_fullscene_*` script imports this module as
`insseg_eval_utils`; when running a method, copy this file next to that script
(the original runs kept a byte-identical copy in each method's `tools/`) or add
`evaluation/` to `PYTHONPATH`.

---

## PointGroup

Pointcept's native PointGroup (`insseg-pointgroup-v1m1`); grouping is run at a
7.5&nbsp;cm radius (facade scale). Pointcept's `v1m1` PointGroup has no ScoreNet
(the proposal score is the mean semantic softmax), no NMS, and no proposal cap.
**Test-set AP:** 0.0145 mAP / 0.0427 AP50 / 0.0955 AP25.

**Upstream:** [Pointcept](https://github.com/Pointcept/Pointcept) (native
PointGroup, not the original spconv-1.0 `dvlab-research/PointGroup`). Clone
Pointcept, overlay the HFX3D files, and apply the patch:

- `configs/insseg-pointgroup-v1m1-0-spunet-fresh.py` → Pointcept `configs/hfx3d/`
- `datasets/hfx3d_inst.py` → Pointcept `pointcept/datasets/`
- `tools/test_fullscene_insseg_hfx3d_fullres.py` → Pointcept `tools/` (plus
  `evaluation/insseg_eval_utils.py` alongside it)
- `pg_fresh_patch.py` — applies the two protocol fixes in place: scale HDF5
  `[0,1]` colour to `0–255` so Pointcept's colour transforms behave as designed,
  and ignore thing points with `instance == -1` instead of merging them into one
  fake instance per crop (`python pg_fresh_patch.py /path/to/Pointcept`).

**Run** ([`slurm/instance/pointgroup/`](../../slurm/instance/pointgroup)):
`pg_train_tamia.sh` (train, 4 GPUs, seed 42) → `pg_select_test_tamia.sh` (or
`pg_select_test_1gpu.sh`: full-scene val AP over every checkpoint, select best,
test once) → `pg_smoke_tamia.sh` for a quick check.

## ISBNet

Trained from scratch following ISBNet's own two-stage recipe (backbone pretrain,
then instance head with the backbone frozen). ISBNet stands in for HAIS, whose
reference repo depends on a dead spconv 1.0 stack. **Test-set AP:** 0.0422 mAP /
0.0883 AP50 / 0.2276 AP25.

**Upstream:** [VinAIResearch/ISBNet](https://github.com/VinAIResearch/ISBNet)
(CVPR'23); the HFX3D configs are modelled on ISBNet's own `configs/stpls3d/`.
`isbnet_fresh_vs_old.diff` is the full diff of every change against upstream.
Overlay onto an ISBNet checkout:

- `configs/{isbnet_backbone_hfx3d.yaml, isbnet_hfx3d.yaml}` → `configs/hfx3d/`
- `data/hfx3d.py` → `isbnet/data/`
- `tools/test_fullscene_hfx3d.py` → `tools/` (plus
  `evaluation/insseg_eval_utils.py` alongside it)
- `isbnet_hfx3d_converter.py` — converts the HFX3D instance HDF5s into ISBNet's
  expected format

**Run** ([`slurm/instance/isbnet/`](../../slurm/instance/isbnet), shared settings
in `common_isbnet_tamia.sh`): convert with `isbnet_hfx3d_converter.py` →
`train_isbnet_fresh_tamia.sh` (stage 1 backbone ~6k steps, stage 2 instance ~12k,
18k total) → `eval_isbnet_fresh_tamia.sh` (val selection then single test) →
`smoke_isbnet_fresh_tamia.sh`. `run_in_container.sh` wraps calls in the ISBNet
Apptainer image.

## SoftGroup

Building-level training and evaluation (not the per-15&nbsp;m-block scoring of
earlier runs), 7-D input, facade-scale soft grouping. SoftGroup is the strongest
learned method on this benchmark. **Test-set AP:** 0.2032 mAP / 0.2923 AP50 /
0.4044 AP25.

**Upstream:** [thangvubk/SoftGroup](https://github.com/thangvubk/SoftGroup)
(commit `a0b5f99`). `review_diff_vs_old.patch` is the full diff of every HFX3D
change against upstream. Overlay onto a SoftGroup checkout:

- `configs/{softgroup_hfx3d.yaml, softgroup_hfx3d_fresh_backbone.yaml,
  softgroup_hfx3d_fresh_full.yaml, softgroup_hfx3d_test.yaml}` → `configs/softgroup/`
- `data/hfx3d.py` → `softgroup/data/`
- `dataset/prepare_data_hfx3d.py` → `dataset/hfx3d/`
- `tools/{train_hfx3d.py, test_fullscene_softgroup_hfx3d.py}` → `tools/` (plus
  `evaluation/insseg_eval_utils.py` alongside the test script)

**Run** ([`slurm/instance/softgroup/`](../../slurm/instance/softgroup)): build
data with `prepare_data_hfx3d.py` → `train_full_tamia.sh` (stage 1 backbone then
stage 2 full, seed 42) → `eval_tamia.sh` (val selection then single test) →
`smoke_tamia.sh`. Seed 43/44 and the earlier per-block "bigblock" study configs
are intentionally omitted (the paper reports seed 42, building-level only).

---

## Geometry clustering baselines (oracle-semantic)

A complementary diagnostic that clusters each thing class's points **given
ground-truth semantics**, isolating instance separation from recognition. Code
is in [`models/clustering/`](../clustering); the four estimators are
scikit-learn's DBSCAN, HDBSCAN, Mean Shift, and K-Means.

| Method | mAP | AP50 | AP25 |
|--------|:---:|:----:|:----:|
| HDBSCAN  | **0.2533** | **0.3430** | **0.4521** |
| DBSCAN   | 0.2063 | 0.2725 | 0.3574 |
| Mean Shift | 0.0310 | 0.0871 | 0.1059 |
| K-Means  | 0.0201 | 0.0615 | 0.1793 |

Each method has one scale parameter tuned on validation; the test split is then
run once with it. Points are clustered on 5 cm voxels (mean 7-D feature per
voxel) standardised with training-split statistics, so a given radius means the
same physical distance in every building; every point takes its voxel's label
(full resolution) and is scored with the same shared evaluator
([`evaluation/insseg_eval_utils.py`](../../evaluation)) as the learned methods.

**Files** ([`models/clustering/`](../clustering)):
- `derive_cluster_hparams.py` — derive every hyperparameter from the training
  split only (writes a JSON the evaluator reads)
- `cluster_common.py` — shared voxelisation / standardisation helpers
- `eval_cluster_hfx3d.py` — cluster one method on a split and score it
- `select_best.py` — pick each method's scale parameter on validation

**Run:** [`slurm/clustering/run_clustering_fir.sh`](../../slurm/clustering) chains
derive → validation sweep → single test (CPU only; needs numpy, scipy,
scikit-learn and h5py).

---

## Reproducibility note

The SLURM scripts under [`slurm/instance/`](../../slurm/instance) are the
**as-run** cluster scripts. They carry cluster-specific paths and identifiers
(`/scratch/.../hfx3d/...`, Apptainer `.sif` images, SLURM accounts, host names)
and are published verbatim for transparency — adapt the paths, container image,
account, and GPU request to your own environment before running. The configs,
loaders, converters, evaluators, and patches are environment-independent.
