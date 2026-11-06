"""
Full-scene instance segmentation evaluation for HFX3D.

Run ONCE, after training, on a chosen checkpoint (typically model_best.pth --
selected by InsSegEvaluator's per-epoch single-random-crop val AP, which
stays exactly as-is for checkpoint selection; that mechanism is cheap and
already wired into CheckpointSaver, see
configs/hfx3d/insseg-pointgroup-v1m1-0-spunet-base.py). This script produces
the protocol's real, whole-building reported metric, which the per-epoch
training signal was never meant to approximate.

Method:
  1. Tile the whole building into overlapping vertical-column tiles (fixed
     XY footprint, full building height kept -- same geometry as the
     ColumnCrop transform used for training/val, just applied systematically
     across the whole footprint instead of once at a random location).
  2. Run PointGroup inference on each tile independently (identical
     preprocessing to the val transform pipeline: CenterShift -> GridSample
     -> NormalizeColor -> InstanceParser, minus ColumnCrop itself, which is
     replaced by this script's own tile selection).
  3. Merge every tile's instance proposals into whole-building instances:
     same-class mask-IoU >= merge-iou proposals are unioned via connected
     components. This single union-merge step does both jobs the protocol
     sketch described as two ("fuse split fragments" + "dedupe repeated
     detections") -- see Engineering Fixes Log for why a second, separate
     NMS-by-suppression pass on top was dropped as redundant (discard-based
     NMS would throw away exactly the completeness gain fusion exists for).
  4. Drop merged instances under --min-inst-points.
  5. Score with the same ScanNet-v2 AP/AP50/AP25 evaluator used everywhere
     else in this project (tools/insseg_eval_utils.py) -- numbers are
     directly comparable to PointGroup's own per-epoch val AP and to the
     clustering-oracle baselines.

Currently implemented for PointGroup only. To extend to Mask3D/HAIS/
SoftGroup: only build_tile_data_dict() (per-model preprocessing) and
infer_tile() (per-model forward + mask/class/score extraction) are
model-specific -- generate_tile_centers/tile_point_indices/
merge_tile_predictions and the evaluator are already model-agnostic.

HOW TO RUN THIS YOURSELF:
    cd Pointcept
    python tools/test_fullscene_insseg_hfx3d.py \\
        --config-file configs/hfx3d/insseg-pointgroup-v1m1-0-spunet-base.py \\
        --options save_path=exp/hfx3d/pointgroup_hfx3d_s42/eval_fullscene_val \\
                  weight=exp/hfx3d/pointgroup_hfx3d_s42/model/model_best.pth \\
                  eval_split=val
Requires a GPU (pointops KNN + the PointGroup/spconv backbone). See
slurm/instance/pointgroup/pg_select_test_tamia.sh for the SLURM version
(runs both val and test).

Useful --options overrides (all optional, defaults shown):
    eval_split=val                  # or test
    stitch_footprint=15.0           # tile XY footprint in metres
    stitch_overlap=5.0              # overlap between adjacent tiles, metres
    stitch_point_max=250000         # per-tile point cap (matches ColumnCrop)
    stitch_merge_iou=0.5            # cross-tile same-class union-merge IoU
    stitch_min_inst_points=100      # drop merged instances smaller than this
"""

import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from insseg_eval_utils import ScanNetInstanceEvaluator

from pointcept.datasets import build_dataset
from pointcept.datasets.transform import (
    CenterShift,
    Collect,
    Copy,
    GridSample,
    InstanceParser,
    NormalizeColor,
    ToTensor,
)
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.launch import launch
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger

try:
    import pointops
except ImportError:
    pointops = None

MIN_TILE_POINTS = 50  # skip tiles this small -- not worth a forward pass


def build_and_load_model(cfg, logger):
    model = build_model(cfg.model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Num params: {n_parameters}")
    model = model.cuda()
    assert os.path.isfile(cfg.weight), f"No checkpoint found at '{cfg.weight}'"
    logger.info(f"Loading weight at: {cfg.weight}")
    checkpoint = torch.load(cfg.weight, weights_only=False)
    weight = {}
    for key, value in checkpoint["state_dict"].items():
        weight[key[7:] if key.startswith("module.") else key] = value
    model.load_state_dict(weight, strict=True)
    logger.info("=> Loaded weight (epoch {})".format(checkpoint["epoch"]))
    return model


def _axis_centers(v_min, v_max, footprint, stride):
    """Tile centres along one axis such that [v_min, v_max] is fully
    covered: first window starts exactly at v_min, last one ends exactly at
    v_max, stepped by `stride` in between (stride <= footprint guarantees no
    gaps -- the whole point of the "overlap" parameter). np.arange's
    exclusive stop can undershoot the last center by construction, so the
    v_max-anchored center is appended explicitly rather than relied upon
    falling out of the range by chance."""
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
    """Grid of tile centres covering the whole XY extent of coord_xy, with
    `overlap` metres of overlap between adjacent tiles."""
    stride = footprint - overlap
    assert stride > 0, "overlap must be smaller than footprint"
    xy_min = coord_xy.min(axis=0)
    xy_max = coord_xy.max(axis=0)
    xs = _axis_centers(xy_min[0], xy_max[0], footprint, stride)
    ys = _axis_centers(xy_min[1], xy_max[1], footprint, stride)
    return [(x, y) for x in xs for y in ys]


def tile_point_indices(coord_xy, center, half_footprint):
    return np.flatnonzero(np.all(np.abs(coord_xy - center) <= half_footprint, axis=1))


def build_tile_data_dict(raw, idx_crop, segment_ignore_index, grid_size):
    """Mirrors the val transform pipeline in
    configs/hfx3d/insseg-pointgroup-v1m1-0-spunet-base.py exactly, minus
    ColumnCrop -- idx_crop (this script's systematic tile selection) takes
    its place."""
    data_dict = {
        "coord": raw["coord"][idx_crop].copy(),
        "color": raw["color"][idx_crop].copy(),
        "normal": raw["normal"][idx_crop].copy(),
        "intensity": raw["intensity"][idx_crop].copy(),
        "segment": raw["segment"][idx_crop].copy(),
        "instance": raw["instance"][idx_crop].copy(),
        "index_valid_keys": [
            "coord", "color", "normal", "intensity", "segment", "instance",
        ],
    }
    data_dict = CenterShift(apply_z=True)(data_dict)
    data_dict = Copy(
        keys_dict={
            "coord": "origin_coord",
            "segment": "origin_segment",
            "instance": "origin_instance",
        }
    )(data_dict)
    data_dict = GridSample(
        grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True
    )(data_dict)
    data_dict = CenterShift(apply_z=False)(data_dict)
    data_dict = NormalizeColor()(data_dict)
    data_dict = InstanceParser(
        segment_ignore_index=segment_ignore_index, instance_ignore_index=-1
    )(data_dict)
    data_dict = ToTensor()(data_dict)
    data_dict = Collect(
        keys=(
            "coord", "grid_coord", "segment", "instance",
            "origin_coord", "origin_segment", "origin_instance",
            "instance_centroid", "bbox",
        ),
        feat_keys=("coord", "color", "intensity"),
        offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
    )(data_dict)
    return data_dict


@torch.no_grad()
def infer_tile(model, data_dict):
    """Runs one tile through the model, returns (pred_masks, pred_classes,
    pred_scores) re-projected onto the tile's pre-voxel ("origin") point
    order -- same KNN reverse-mapping InsSegTester.test() uses."""
    n_origin = data_dict["origin_coord"].shape[0]
    for key in list(data_dict.keys()):
        if isinstance(data_dict[key], torch.Tensor):
            data_dict[key] = data_dict[key].cuda(non_blocking=True)

    output_dict = model(data_dict)
    if output_dict["pred_classes"].shape[0] == 0:
        return (
            np.zeros((0, n_origin), dtype=bool),
            np.zeros((0), dtype=int),
            np.zeros((0), dtype=float),
        )

    reverse, _ = pointops.knn_query(
        1,
        data_dict["coord"].float(),
        data_dict["offset"].int(),
        data_dict["origin_coord"].float(),
        data_dict["origin_offset"].int(),
    )
    reverse = reverse.cpu().flatten().long()
    pred_masks = output_dict["pred_masks"][:, reverse].numpy().astype(bool)
    pred_classes = output_dict["pred_classes"].numpy().astype(int)
    pred_scores = output_dict["pred_scores"].numpy().astype(float)
    return pred_masks, pred_classes, pred_scores


def merge_tile_predictions(n_points_building, tile_results, merge_iou_thresh, min_inst_points):
    """tile_results: list of (idx_crop, pred_masks, pred_classes, pred_scores)
    where idx_crop holds GLOBAL (whole-building) point indices, in the same
    order pred_masks' columns are in. Returns (masks, classes, scores) in
    global point-index space, one entry per merged instance."""
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


def process_building(
    model, raw, segment_ignore_index, footprint, overlap, point_max,
    grid_size, merge_iou, min_inst_points, logger, seed,
):
    coord = raw["coord"]
    n_points = coord.shape[0]
    coord_xy = coord[:, :2]
    centers = generate_tile_centers(coord_xy, footprint, overlap)
    half_footprint = footprint / 2

    # A fixed, per-point priority computed ONCE for the whole building, not a
    # per-tile call into one evolving RandomState. This is the root cause of
    # PointGroup's near-zero cross-tile merge rate: 69-79% of tiles get capped
    # at point_max in every building (e.g. BLD008 10/14, BLD022 45/65,
    # BLD025 15/19), and the OLD
    # code called `rng.choice(idx_crop, point_max, replace=False)` on the
    # SAME evolving RandomState once per tile -- so two overlapping tiles
    # that both need capping draw two independent, essentially uncorrelated
    # random subsets of their shared overlap region. Even when PointGroup
    # correctly detects the SAME real object in both tiles, their point-index
    # masks then barely intersect (different points sampled), so cross-tile
    # mask-IoU (which is computed on point-index overlap) almost never
    # clears the 0.5 threshold -- consistent with the observed pattern
    # (high-capping buildings <-> near-zero merge rate, uniformly). This is a
    # bug in this tiling/merge script, not in PointGroup's model.
    #
    # Fix: rank every point in the building ONCE via a fixed permutation,
    # then within any tile that needs capping, keep the point_max
    # LOWEST-ranked (by this fixed value) points among that tile's
    # candidates. A point's rank never changes between tiles, so two
    # overlapping tiles largely agree on which shared points survive --
    # not a perfect guarantee (the exact keep/drop cutoff still depends on
    # each tile's own total candidate count), but a large, principled
    # improvement over independent random draws, which had no correlation
    # at all.
    point_priority = np.random.RandomState(seed).permutation(n_points)

    tile_results = []
    n_tiles_used = 0
    n_tiles_capped = 0
    n_raw_proposals = 0
    for tile_i, center in enumerate(centers):
        np.random.seed((seed * 100003 + tile_i) % (2 ** 32))  # deterministic GridSample per tile
        idx_full = tile_point_indices(coord_xy, np.array(center), half_footprint)
        if idx_full.shape[0] < MIN_TILE_POINTS:
            continue
        idx_crop = idx_full
        if idx_full.shape[0] > point_max:
            local_priority = point_priority[idx_full]
            keep = np.argpartition(local_priority, point_max)[:point_max]
            idx_crop = idx_full[keep]
            n_tiles_capped += 1
        tile_data = build_tile_data_dict(raw, idx_crop, segment_ignore_index, grid_size)
        pred_masks, pred_classes, pred_scores = infer_tile(model, tile_data)
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


def main_worker(cfg):
    cfg = default_setup(cfg)
    logger = get_root_logger(
        log_file=os.path.join(cfg.save_path, "test_fullscene.log"),
        file_mode="a" if cfg.resume else "w",
    )
    logger.info("=> Full-scene instance-seg evaluation [FULLRES: masks propagated to all tile points]")
    logger.info(f"Config:\n{cfg.pretty_text}")

    model = build_and_load_model(cfg, logger)
    model.eval()

    split = cfg.get("eval_split", "val")
    footprint = cfg.get("stitch_footprint", 15.0)
    overlap = cfg.get("stitch_overlap", 5.0)
    point_max = cfg.get("stitch_point_max", 250000)
    merge_iou = cfg.get("stitch_merge_iou", 0.5)
    min_inst_points = cfg.get("stitch_min_inst_points", 100)
    grid_size = cfg.model.voxel_size

    logger.info(
        f"split={split} footprint={footprint} overlap={overlap} "
        f"point_max={point_max} merge_iou={merge_iou} min_inst_points={min_inst_points}"
    )

    raw_dataset = build_dataset(
        dict(
            type=cfg.dataset_type,
            split=split,
            data_root=cfg.data.val.data_root,
            transform=[],
        )
    )

    evaluator = ScanNetInstanceEvaluator(
        cfg.data.names, cfg.model.segment_ignore_index, -1
    )
    scenes = []
    n_buildings = len(raw_dataset.data_list)
    for idx in range(n_buildings):
        name = raw_dataset.get_data_name(idx)
        t0 = time.time()
        raw = raw_dataset.get_data(idx)
        logger.info(f"[{idx + 1}/{n_buildings}] {name}: {raw['coord'].shape[0]} points")
        final_masks, final_classes, final_scores = process_building(
            model, raw, cfg.model.segment_ignore_index, footprint, overlap,
            point_max, grid_size, merge_iou, min_inst_points, logger, seed=42 + idx,
        )
        # qualdump save stitched per-point instance predictions (additive) ---
        import os as _os_qd, numpy as _np_qd
        _pred_dir = _os_qd.path.join(cfg.save_path, "predictions")
        _os_qd.makedirs(_pred_dir, exist_ok=True)
        _coord_qd = raw["coord"]
        _inst_qd = _np_qd.full(_coord_qd.shape[0], -1, dtype=_np_qd.int32)
        _ord_qd = _np_qd.argsort(_np_qd.asarray(final_scores)) if len(final_scores) else []
        for _i_qd in _ord_qd:
            _inst_qd[final_masks[int(_i_qd)]] = int(_i_qd)
        _np_qd.save(_os_qd.path.join(_pred_dir, str(name) + "_instance.npy"), _inst_qd)
        _np_qd.save(_os_qd.path.join(_pred_dir, str(name) + "_xyz.npy"), _np_qd.asarray(_coord_qd, dtype=_np_qd.float32))
        _np_qd.save(_os_qd.path.join(_pred_dir, str(name) + "_classes.npy"), _np_qd.asarray(final_classes, dtype=_np_qd.int32))
        _np_qd.save(_os_qd.path.join(_pred_dir, str(name) + "_scores.npy"), _np_qd.asarray(final_scores, dtype=_np_qd.float32))
        # qualdump end ---
        gt_instances, pred_instances = evaluator.associate_instances(
            final_masks, final_classes, final_scores, raw["segment"], raw["instance"]
        )
        scenes.append(dict(gt=gt_instances, pred=pred_instances))
        logger.info(f"  done in {time.time() - t0:.1f}s")

    ap_scores = evaluator.evaluate_matches(scenes)
    logger.info(
        "=== Full-scene {} result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f} ===".format(
            split, ap_scores["all_ap"], ap_scores["all_ap_50"], ap_scores["all_ap_25"]
        )
    )
    for label_name in evaluator.valid_class_names:
        c = ap_scores["classes"][label_name]
        logger.info(
            f"  {label_name:12s} AP/AP50/AP25: {c['ap']:.4f}/{c['ap50']:.4f}/{c['ap25']:.4f}"
        )


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg),
    )


if __name__ == "__main__":
    main()
