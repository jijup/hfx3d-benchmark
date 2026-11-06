"""
Full-scene instance-segmentation evaluation of ISBNet on HFX3D.

Tiling, forward-pass subsampling, 1-NN propagation, cross-tile merge, size
filter and scoring are copied from the PointGroup reference script
(models/instance/pointgroup/tools/test_fullscene_insseg_hfx3d_fullres.py) so
both methods are scored by the same procedure:
  1. 15 m x 15 m full-height tiles, 5 m overlap (_axis_centers /
     generate_tile_centers / tile_point_indices: verbatim).
  2. A tile with >250k points keeps the 250k points with the lowest value of
     ONE seeded permutation of the building (seed = 42 + building index in the
     glob order of the split directory, as in the PointGroup script), so
     overlapping tiles keep the same points.
  3. Every predicted mask is propagated from the forward-pass points to EVERY
     point of the tile by 1-NN (scipy cKDTree on xyz) BEFORE merging. This is
     the fix for the audit's subsample bug: the old ISBNet script kept
     predictions only on the <=250k forward-pass points and scored them
     against full-resolution GT.
  4. Same-class proposals with mask IoU >= 0.5 are unioned (connected
     components), merged score = max member score (merge_tile_predictions:
     verbatim).
  5. Merged predictions with < 100 points are discarded.
  6. ScanNet-v2 evaluator: tools/insseg_eval_utils.py is a byte-identical
     copy of hfx3d-benchmark/evaluation/insseg_eval_utils.py (void = stuff
     classes OR instance == -1). GT is read straight from the released .h5
     (labels/semantic raw 0-11, labels/instance with -1 = ignore), with the
     same class list and segment_ignore_index (-1, 0, 6, 7, 9) as PointGroup.

ISBNet-specific parts (only these differ from the PointGroup script):
  * infer_tile(): tile-local coordinates (HFX3DDataset's tile_local_coords,
    the same shift used for training crops), ISBNet's own transform_test
    (incl. its fixed 0.35*pi rotation) and collate_fn, model forward, RLE
    decode. GT labels are NOT passed to the model (dummy labels).
  * Predicted classes are ISBNet thing indices 0..7 and are mapped back to
    raw HFX3D ids (window 1, door 2, balcony 3, vegetation 4, stairs 5,
    blinds 8, column 10, arch 11). Score = ISBNet's native
    sqrt(cls_prob * mask_iou_conf).
  * Inference is deterministic (no random op at test time in ISBNet's
    forward_test with iterative_sampling False; the fixed permutation is
    seeded).

HOW TO RUN (inside the ISBNet container, from the ISBNet repo root):
    export PYTHONPATH=$PWD:$PWD/isbnet/pointnet2:$PWD/.deps
    python3 tools/test_fullscene_hfx3d.py configs/hfx3d/isbnet_hfx3d.yaml \\
        work_dirs/instance_s42/step_18000.pth \\
        --data_root /scratch/abdiaziz/hfx3d/fresh/data/HFX3D_Instance \\
        --eval_split val --save_dir work_dirs/instance_s42/eval_val_step_18000
    Optional: --only HFX_BLD008_ZEB_CLEAN (score one building; its seed is
    still taken from its position in the full split directory listing).
See slurm/eval_isbnet_fresh_tamia.sh for the SLURM version.
"""

import argparse
import glob
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
import yaml
from munch import Munch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from insseg_eval_utils import ScanNetInstanceEvaluator  # noqa: E402

from isbnet.data.hfx3d import HFX3DDataset, tile_local_coords  # noqa: E402
from isbnet.model import ISBNet  # noqa: E402
from isbnet.util import get_root_logger, load_checkpoint, rle_decode  # noqa: E402

# Raw HFX3D label order and stuff/ignore set -- identical to
# configs/hfx3d/insseg-pointgroup-v1m1-0-spunet-base.py (class_names,
# segment_ignore_index), which the PointGroup full-scene script passes to the
# same evaluator.
CLASS_NAMES = [
    "wall", "window", "door", "balcony", "vegetation", "stairs",
    "terrain", "roof", "blinds", "other", "column", "arch",
]
SEGMENT_IGNORE_INDEX = (-1, 0, 6, 7, 9)
INSTANCE_IGNORE_INDEX = -1
# ISBNet thing index (dense id - 4, see isbnet_hfx3d_converter.py) -> raw HFX3D id
THING_TO_RAW = np.array([1, 2, 3, 4, 5, 8, 10, 11], dtype=int)
MIN_TILE_POINTS = 50  # skip tiles this small -- not worth a forward pass (as PointGroup)


def get_args():
    parser = argparse.ArgumentParser("ISBNet HFX3D full-scene evaluation")
    parser.add_argument("config", type=str)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--data_root", type=str, required=True, help="HFX3D_Instance root with <split>/*.h5")
    parser.add_argument("--eval_split", type=str, default="val")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--only", type=str, nargs="*", default=None, help="building names to score (subset)")
    parser.add_argument("--footprint", type=float, default=15.0)
    parser.add_argument("--overlap", type=float, default=5.0)
    parser.add_argument("--point_max", type=int, default=250000)
    parser.add_argument("--merge_iou", type=float, default=0.5)
    parser.add_argument("--min_inst_points", type=int, default=100)
    parser.add_argument("--no_dump", action="store_true", help="do not save per-point predictions")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model-agnostic part: verbatim from test_fullscene_insseg_hfx3d_fullres.py
# ---------------------------------------------------------------------------
def _axis_centers(v_min, v_max, footprint, stride):
    half = footprint / 2
    if v_max - v_min <= footprint:
        return [(v_min + v_max) / 2]
    centers = list(np.arange(v_min + half, v_max - half, stride))
    if not centers:
        centers = [v_min + half]
    if centers[-1] < v_max - half - 1e-6:
        centers.append(v_max - half)
    return centers


def generate_tile_centers(coord_xy, footprint, overlap):
    stride = footprint - overlap
    assert stride > 0, "overlap must be smaller than footprint"
    xy_min = coord_xy.min(axis=0)
    xy_max = coord_xy.max(axis=0)
    xs = _axis_centers(xy_min[0], xy_max[0], footprint, stride)
    ys = _axis_centers(xy_min[1], xy_max[1], footprint, stride)
    return [(x, y) for x in xs for y in ys]


def tile_point_indices(coord_xy, center, half_footprint):
    return np.flatnonzero(np.all(np.abs(coord_xy - center) <= half_footprint, axis=1))


def merge_tile_predictions(n_points_building, tile_results, merge_iou_thresh, min_inst_points):
    per_class = {}
    for idx_crop, pred_masks, pred_classes, pred_scores in tile_results:
        for m, c, s in zip(pred_masks, pred_classes, pred_scores):
            global_idx = np.sort(idx_crop[m])
            if global_idx.shape[0] == 0:
                continue
            per_class.setdefault(int(c), []).append(dict(idx=global_idx, score=float(s)))

    final_masks, final_classes, final_scores = [], [], []
    for class_id, proposals in per_class.items():
        n = len(proposals)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                inter = np.intersect1d(
                    proposals[i]["idx"], proposals[j]["idx"], assume_unique=True
                ).shape[0]
                if inter == 0:
                    continue
                union_size = (
                    proposals[i]["idx"].shape[0] + proposals[j]["idx"].shape[0] - inter
                )
                if inter / union_size >= merge_iou_thresh:
                    union(i, j)

        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        for members in groups.values():
            merged_idx = np.unique(np.concatenate([proposals[i]["idx"] for i in members]))
            if merged_idx.shape[0] < min_inst_points:
                continue
            mask = np.zeros(n_points_building, dtype=bool)
            mask[merged_idx] = True
            final_masks.append(mask)
            final_classes.append(class_id)
            final_scores.append(max(proposals[i]["score"] for i in members))

    return final_masks, final_classes, final_scores


# ---------------------------------------------------------------------------
# ISBNet-specific part
# ---------------------------------------------------------------------------
@torch.no_grad()
def infer_tile(model, dataset, scan_id, xyz, feat):
    """One forward pass on the given (<= point_max) points. Returns
    (pred_masks[bool, n_pts], pred_classes[raw HFX3D id], pred_scores) in the
    order of the input points."""
    n = xyz.shape[0]
    xyz_local = tile_local_coords(xyz.astype(np.float32))
    dummy_sem = np.zeros(n, dtype=np.int64)  # GT is never given to the model
    dummy_inst = np.full(n, -100, dtype=np.int64)
    xyz_t, xyz_middle, feat_t, sem_t, inst_t, spp_t = dataset.transform_test(
        xyz_local, feat.astype(np.float32), dummy_sem, dummy_inst, np.arange(n, dtype=np.int64)
    )
    item = (
        scan_id,
        torch.from_numpy(xyz_t).long(),
        torch.from_numpy(xyz_middle),
        torch.from_numpy(feat_t).float(),
        torch.from_numpy(sem_t),
        torch.from_numpy(inst_t),
        torch.from_numpy(spp_t),
        0,
    )
    batch = dataset.collate_fn([item])
    for key in list(batch.keys()):
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].cuda(non_blocking=True)

    res = model(batch)
    pred_instances = res.get("pred_instances", [])
    if len(pred_instances) == 0:
        return np.zeros((0, n), dtype=bool), np.zeros((0), dtype=int), np.zeros((0), dtype=float)
    masks = np.stack([rle_decode(p["pred_mask"]) for p in pred_instances]).astype(bool)
    assert masks.shape[1] == n
    # get_instance() hfx3d branch: label_id = thing index + 1
    classes = THING_TO_RAW[np.array([p["label_id"] - 1 for p in pred_instances], dtype=int)]
    scores = np.array([p["conf"] for p in pred_instances], dtype=float)
    return masks, classes, scores


def process_building(model, dataset, name, coord, feat, footprint, overlap, point_max,
                     merge_iou, min_inst_points, logger, seed):
    n_points = coord.shape[0]
    coord_xy = coord[:, :2]
    centers = generate_tile_centers(coord_xy, footprint, overlap)
    half_footprint = footprint / 2
    # Fixed per-point priority for the whole building (same as PointGroup).
    point_priority = np.random.RandomState(seed).permutation(n_points)

    tile_results = []
    n_tiles_used = 0
    n_tiles_capped = 0
    n_raw_proposals = 0
    for center in centers:
        idx_full = tile_point_indices(coord_xy, np.array(center), half_footprint)
        if idx_full.shape[0] < MIN_TILE_POINTS:
            continue
        idx_crop = idx_full
        if idx_full.shape[0] > point_max:
            local_priority = point_priority[idx_full]
            keep = np.argpartition(local_priority, point_max)[:point_max]
            idx_crop = idx_full[keep]
            n_tiles_capped += 1
        pred_masks, pred_classes, pred_scores = infer_tile(model, dataset, name, coord[idx_crop], feat[idx_crop])
        n_tiles_used += 1
        n_raw_proposals += pred_masks.shape[0]
        if pred_masks.shape[0] > 0:
            if idx_crop.shape[0] != idx_full.shape[0]:
                # FULLRES FIX: give every tile point the prediction of its
                # nearest subsampled point, so masks are scored at full resolution.
                _, nn = cKDTree(coord[idx_crop]).query(coord[idx_full], k=1, workers=-1)
                pred_masks = pred_masks[:, nn]
            tile_results.append((idx_full, pred_masks, pred_classes, pred_scores))

    logger.info(
        f"  {n_tiles_used} tiles processed ({n_tiles_capped} subsampled at "
        f"point_max={point_max}), {n_raw_proposals} raw proposals"
    )
    final_masks, final_classes, final_scores = merge_tile_predictions(
        n_points, tile_results, merge_iou, min_inst_points
    )
    logger.info(f"  -> {len(final_masks)} merged instances after union-merge + size filter")
    return final_masks, final_classes, final_scores


def load_building(h5_path):
    with h5py.File(h5_path, "r") as f:
        xyz = f["point_cloud/xyz"][:].astype(np.float32)
        rgb = f["point_cloud/rgb"][:].astype(np.float32)
        intensity = f["point_cloud/intensity"][:].astype(np.float32)
        segment = f["labels/semantic"][:].reshape([-1]).astype(np.int32)
        instance = f["labels/instance"][:].reshape([-1]).astype(np.int32)
    # Same 4-channel feature the converter writes for training: released [0,1] scales.
    feat = np.concatenate([rgb, intensity.reshape(-1, 1)], axis=1)
    return xyz, feat, segment, instance


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    logger = get_root_logger(log_file=os.path.join(args.save_dir, "test_fullscene.log"))
    logger.info(f"ISBNet full-scene eval. args={vars(args)}")

    cfg = Munch.fromDict(yaml.safe_load(open(args.config).read()))
    assert not cfg.model.semantic_only
    model = ISBNet(**cfg.model, dataset_name="hfx3d").cuda()
    model.eval()
    logger.info(f"Loading weight at: {args.checkpoint}")
    load_checkpoint(args.checkpoint, logger, model)
    ck = torch.load(args.checkpoint, map_location="cpu")
    logger.info(f"checkpoint global_step={ck.get('global_step')} stage_step={ck.get('step')}")

    # Used only for its transform_test/collate_fn/voxel_cfg (no file list is read).
    dataset = HFX3DDataset.__new__(HFX3DDataset)
    dataset.voxel_cfg = Munch.fromDict(cfg.data.test.voxel_cfg)
    dataset.training = False
    dataset.logger = logger

    evaluator = ScanNetInstanceEvaluator(CLASS_NAMES, SEGMENT_IGNORE_INDEX, INSTANCE_IGNORE_INDEX)
    # Same building order / seed rule as the PointGroup script: Pointcept's
    # DefaultDataset.get_data_list() is an UNSORTED glob of <split>/*, and the
    # seed is 42 + index in that list.
    data_list = glob.glob(os.path.join(args.data_root, args.eval_split, "*"))
    scenes = []
    per_building = {}
    t_all = time.time()
    for idx, path in enumerate(data_list):
        name = os.path.splitext(os.path.basename(path))[0]
        if args.only and name not in args.only:
            continue
        t0 = time.time()
        coord, feat, segment, instance = load_building(path)
        logger.info(f"[{idx + 1}/{len(data_list)}] {name}: {coord.shape[0]} points (seed {42 + idx})")
        final_masks, final_classes, final_scores = process_building(
            model, dataset, name, coord, feat, args.footprint, args.overlap, args.point_max,
            args.merge_iou, args.min_inst_points, logger, seed=42 + idx,
        )
        if not args.no_dump:
            pred_dir = os.path.join(args.save_dir, "predictions")
            os.makedirs(pred_dir, exist_ok=True)
            inst = np.full(coord.shape[0], -1, dtype=np.int32)
            for i in (np.argsort(np.asarray(final_scores)) if len(final_scores) else []):
                inst[final_masks[int(i)]] = int(i)
            np.save(os.path.join(pred_dir, f"{name}_instance.npy"), inst)
            np.save(os.path.join(pred_dir, f"{name}_classes.npy"), np.asarray(final_classes, dtype=np.int32))
            np.save(os.path.join(pred_dir, f"{name}_scores.npy"), np.asarray(final_scores, dtype=np.float32))
        gt_instances, pred_instances = evaluator.associate_instances(
            final_masks, final_classes, final_scores, segment, instance
        )
        scenes.append(dict(gt=gt_instances, pred=pred_instances))
        per_building[name] = dict(n_points=int(coord.shape[0]), n_pred=len(final_masks), seed=42 + idx,
                                  seconds=round(time.time() - t0, 1))
        logger.info(f"  done in {time.time() - t0:.1f}s")

    assert scenes, "no building evaluated"
    ap_scores = evaluator.evaluate_matches(scenes)
    logger.info(
        "=== Full-scene {} result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f} ===".format(
            args.eval_split, ap_scores["all_ap"], ap_scores["all_ap_50"], ap_scores["all_ap_25"]
        )
    )
    for label_name in evaluator.valid_class_names:
        c = ap_scores["classes"][label_name]
        logger.info(f"  {label_name:12s} AP/AP50/AP25: {c['ap']:.4f}/{c['ap50']:.4f}/{c['ap25']:.4f}")

    def _f(x):
        return None if x is None or np.isnan(x) else float(x)

    out = dict(
        checkpoint=args.checkpoint, global_step=ck.get("global_step"), split=args.eval_split,
        buildings=per_building, seconds=round(time.time() - t_all, 1),
        AP=_f(ap_scores["all_ap"]), AP50=_f(ap_scores["all_ap_50"]), AP25=_f(ap_scores["all_ap_25"]),
        classes={k: {m: _f(v[m]) for m in ("ap", "ap50", "ap25")} for k, v in ap_scores["classes"].items()},
    )
    with open(os.path.join(args.save_dir, "result.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
