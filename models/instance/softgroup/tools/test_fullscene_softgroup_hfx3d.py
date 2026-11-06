"""Full-scene HFX3D instance evaluation for SoftGroup.

Port of the PointGroup reference
  models/instance/pointgroup/tools/test_fullscene_insseg_hfx3d_fullres.py
Only the two model-specific pieces are new: build_tile_data_dict() and
infer_tile(). Tiling (generate_tile_centers/_axis_centers/tile_point_indices),
the fixed per-building point priority (RandomState(42 + building index in the
sorted split list).permutation), the 250k cap via argpartition, 1-NN
propagation of every mask to every tile point BEFORE merging, the same-class
IoU>=0.5 union-find merge (max member score) and the <100-point filter follow
the reference line for line. merge_tile_predictions() finds candidate pairs
with a sparse intersection matrix instead of testing all O(n^2) pairs; the
pairs it unions and the resulting groups/order are identical to the reference
loop (pairs with zero intersection can never reach IoU>=0.5).

Scoring: tools/insseg_eval_utils.py = byte-identical copy of the repo's
evaluation/insseg_eval_utils.py (ScanNet-v2 port, void = stuff OR instance -1),
per building at full resolution against the released h5 GT.
SoftGroup's own per-block ScanNetEval is not used.

A secondary, NON-protocol diagnostic is also logged: full-resolution semantic
mIoU of the backbone (1-NN-propagated per tile, later tiles overwrite earlier
ones). It is the only signal available for semantic-only (stage 1) checkpoints.

HOW TO RUN (inside the container, repo root as cwd, GPU required):
  python tools/test_fullscene_softgroup_hfx3d.py \
      --config configs/softgroup/softgroup_hfx3d_fresh.yaml \
      --ckpt work_dirs/hfx3d_fresh/step_18000.pth --split val \
      --out_dir work_dirs/hfx3d_fresh/eval_val_step_18000 [--buildings HFX_BLD008_ZEB_CLEAN]
"""
import argparse
import json
import logging
import os
import os.path as osp
import sys
import time

import numpy as np
import scipy.sparse as sp
import spconv.pytorch as spconv
import torch
import yaml
from munch import Munch
from scipy.spatial import cKDTree

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from insseg_eval_utils import ScanNetInstanceEvaluator  # noqa: E402

from softgroup.data.hfx3d import (DENSE_TO_RAW, RAW_CLASS_NAMES, THING_RAW, center_xy,  # noqa
                                  dense_semantic, ensure_cache, load_building, make_sample,
                                  split_building_names)
from softgroup.model import SoftGroup  # noqa: E402
from softgroup.ops import voxelization, voxelization_idx  # noqa: E402
from softgroup.util import rle_decode  # noqa: E402

MIN_TILE_POINTS = 50  # reference: skip tiles this small
SEGMENT_IGNORE_INDEX = (-1, 0, 6, 7, 9)  # stuff (raw ids) -- same as PointGroup eval


# ------------------------------------------------------------------ tiling (reference, verbatim)
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


# ------------------------------------------------------------------ model-specific
def build_tile_data_dict(raw, idx_crop, scale, min_spatial_shape):
    """SoftGroup test-time preprocessing of one tile: same crop-local frame as
    training (xy centred on bbox centre, z as released), 4 feats rgb|intensity
    (+xyz appended by the model, with_coords=True), voxel coords at `scale`,
    no augmentation."""
    xyz_middle = center_xy(raw["xyz"][idx_crop])
    feat = np.asarray(raw["feat"][idx_crop], dtype=np.float32)
    n = xyz_middle.shape[0]
    sem_dummy = np.zeros(n, dtype=np.int64)  # labels are not used at inference
    inst_dummy = np.full(n, -100, dtype=np.int64)
    (_, coord, coord_float, feat_t, _, _, _, _, _, _) = make_sample(
        "tile", xyz_middle, feat, sem_dummy, inst_dummy, scale)
    coords = torch.cat([coord.new_zeros((n, 1)), coord], 1)
    voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, 1)
    spatial_shape = np.clip(coords.max(0)[0][1:].numpy() + 1, min_spatial_shape, None)
    return dict(batch_idxs=coords[:, 0].int(), voxel_coords=voxel_coords, p2v_map=p2v_map,
                v2p_map=v2p_map, coords_float=coord_float, feats=feat_t,
                spatial_shape=spatial_shape, n=n)


@torch.no_grad()
def infer_tile(model, d):
    """Returns (pred_masks [M, n] bool over the forward-pass points,
    pred_classes [M] RAW HFX3D ids, pred_scores [M], sem_pred [n] RAW ids).
    Same computation as SoftGroup.forward_test (no x4_split, no lvl_fusion),
    without its GT bookkeeping."""
    feats = d["feats"].cuda()
    coords_float = d["coords_float"].cuda()
    batch_idxs = d["batch_idxs"].cuda()
    if model.with_coords:
        feats = torch.cat((feats, coords_float), 1)
    voxel_feats = voxelization(feats, d["p2v_map"].cuda())
    inp = spconv.SparseConvTensor(voxel_feats, d["voxel_coords"].int().cuda(), d["spatial_shape"], 1)
    semantic_scores, pt_offsets, output_feats = model.forward_backbone(inp, d["v2p_map"].cuda())
    sem_pred = DENSE_TO_RAW[semantic_scores.max(1)[1].cpu().numpy()]
    n = d["n"]
    empty = (np.zeros((0, n), dtype=bool), np.zeros((0), dtype=int), np.zeros((0), dtype=float))
    if model.semantic_only:
        return (*empty, sem_pred)
    proposals_idx, proposals_offset = model.forward_grouping(
        semantic_scores, pt_offsets, batch_idxs, coords_float, model.grouping_cfg)
    inst_feats, inst_map = model.clusters_voxelization(
        proposals_idx, proposals_offset, output_feats, coords_float, **model.instance_voxel_cfg)
    _, cls_scores, iou_scores, mask_scores = model.forward_instance(inst_feats, inst_map)
    preds = model.get_instances("tile", proposals_idx, semantic_scores, cls_scores, iou_scores,
                                mask_scores)
    if len(preds) == 0:
        return (*empty, sem_pred)
    pred_masks = np.stack([rle_decode(p["pred_mask"]).astype(bool) for p in preds])
    pred_classes = THING_RAW[np.array([int(p["label_id"]) - 1 for p in preds])]
    pred_scores = np.array([float(p["conf"]) for p in preds], dtype=float)
    return pred_masks, pred_classes.astype(int), pred_scores, sem_pred


# ------------------------------------------------------------------ merge (reference semantics)
def merge_tile_predictions(n_points_building, tile_results, merge_iou_thresh, min_inst_points):
    """tile_results: list of (idx_full, pred_masks, pred_classes, pred_scores)
    with masks over idx_full (global point indices). Returns (list of sorted
    global index arrays, classes, scores) -- one per merged instance, in the
    same order as the reference implementation."""
    per_class = {}
    for idx_crop, pred_masks, pred_classes, pred_scores in tile_results:
        for m, c, s in zip(pred_masks, pred_classes, pred_scores):
            global_idx = np.sort(idx_crop[m])
            if global_idx.shape[0] == 0:
                continue
            per_class.setdefault(int(c), []).append(dict(idx=global_idx, score=float(s)))

    final_idx, final_classes, final_scores = [], [], []
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

        sizes = np.array([p["idx"].shape[0] for p in proposals], dtype=np.int64)
        rows = np.repeat(np.arange(n), sizes)
        cols = np.concatenate([p["idx"] for p in proposals])
        A = sp.csr_matrix((np.ones(cols.shape[0], dtype=np.int64), (rows, cols)),
                          shape=(n, n_points_building))
        inter = sp.triu(A @ A.T, k=1).tocoo()  # only pairs i<j that share >=1 point
        order = np.lexsort((inter.col, inter.row))  # reference visits (i, j) in i-major order
        for i, j, v in zip(inter.row[order], inter.col[order], inter.data[order]):
            if v / (sizes[i] + sizes[j] - v) >= merge_iou_thresh:
                union(int(i), int(j))

        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        for members in groups.values():
            merged_idx = np.unique(np.concatenate([proposals[i]["idx"] for i in members]))
            if merged_idx.shape[0] < min_inst_points:
                continue
            final_idx.append(merged_idx)
            final_classes.append(class_id)
            final_scores.append(max(proposals[i]["score"] for i in members))
    return final_idx, final_classes, final_scores


class LazyMasks:
    """Dense boolean masks built one at a time on access. The shared evaluator
    only does np.not_equal(pred_masks[i], 0) per prediction, so this gives the
    exact same result as a list of dense masks without holding them all
    (a 26M-point building x thousands of predictions would not fit in RAM)."""

    def __init__(self, idx_list, n):
        self.idx_list, self.n = idx_list, n

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, i):
        m = np.zeros(self.n, dtype=bool)
        m[self.idx_list[i]] = True
        return m


def process_building(model, raw, footprint, overlap, point_max, scale, min_spatial_shape,
                     merge_iou, min_inst_points, logger, seed):
    coord = np.asarray(raw["xyz"])
    n_points = coord.shape[0]
    coord_xy = coord[:, :2]
    centers = generate_tile_centers(coord_xy, footprint, overlap)
    half_footprint = footprint / 2
    point_priority = np.random.RandomState(seed).permutation(n_points)  # reference
    sem_full = np.full(n_points, -1, dtype=np.int16)

    tile_results = []
    n_tiles_used = n_tiles_capped = n_raw_proposals = 0
    t_fwd = 0.0
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
        t0 = time.time()
        tile = build_tile_data_dict(raw, idx_crop, scale, min_spatial_shape)
        pred_masks, pred_classes, pred_scores, sem_pred = infer_tile(model, tile)
        t_fwd += time.time() - t0
        n_tiles_used += 1
        n_raw_proposals += pred_masks.shape[0]
        nn = None
        if idx_crop.shape[0] != idx_full.shape[0]:
            # FULLRES: every tile point takes the prediction of its nearest forward-pass point
            _, nn = cKDTree(coord[idx_crop]).query(coord[idx_full], k=1, workers=-1)
        sem_full[idx_full] = sem_pred if nn is None else sem_pred[nn]
        if pred_masks.shape[0] > 0:
            if nn is not None:
                pred_masks = pred_masks[:, nn]
            tile_results.append((idx_full, pred_masks, pred_classes, pred_scores))

    logger.info(f"  {n_tiles_used} tiles processed ({n_tiles_capped} subsampled at "
                f"point_max={point_max}), {n_raw_proposals} raw proposals, "
                f"model+preproc {t_fwd:.1f}s")
    t0 = time.time()
    final_idx, final_classes, final_scores = merge_tile_predictions(
        n_points, tile_results, merge_iou, min_inst_points)
    logger.info(f"  -> {len(final_idx)} merged instances after union-merge + size filter "
                f"({time.time() - t0:.1f}s)")
    return final_idx, final_classes, final_scores, sem_full


def build_model(cfg_path, ckpt, logger):
    cfg = Munch.fromDict(yaml.safe_load(open(cfg_path)))
    model = SoftGroup(**cfg.model).cuda()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = {k[7:] if k.startswith("module.") else k: v for k, v in state["net"].items()}
    model.load_state_dict(net, strict=True)
    model.eval()
    logger.info(f"loaded {ckpt} (global_step {state.get('global_step')}, stage "
                f"{state.get('stage')}, semantic_only={model.semantic_only})")
    return cfg, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="model config the checkpoint was trained with")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--buildings", nargs="*", default=None, help="subset (names); default all")
    ap.add_argument("--data_root", default="/scratch/abdiaziz/hfx3d/fresh/softgroup/data/HFX3D_Instance")
    ap.add_argument("--cache_root", default="/scratch/abdiaziz/hfx3d/fresh/softgroup/data/npy_cache")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--footprint", type=float, default=15.0)
    ap.add_argument("--overlap", type=float, default=5.0)
    ap.add_argument("--point_max", type=int, default=250000)
    ap.add_argument("--merge_iou", type=float, default=0.5)
    ap.add_argument("--min_inst_points", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(osp.join(args.out_dir, "eval.log")),
                                  logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger("fullscene")
    logger.info(f"args: {vars(args)}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg, model = build_model(args.config, args.ckpt, logger)
    scale = cfg.data.train.voxel_cfg.scale
    min_spatial_shape = cfg.data.train.voxel_cfg.spatial_shape[0]

    all_names = split_building_names(args.data_root, args.split)  # sorted -> seed index
    names = all_names if not args.buildings else [n for n in all_names if n in args.buildings]
    assert names, "no buildings selected"
    cache = dict(ensure_cache(args.data_root, args.split, args.cache_root, names=names))

    evaluator = ScanNetInstanceEvaluator(list(RAW_CLASS_NAMES), SEGMENT_IGNORE_INDEX, -1)
    scenes, per_building = [], {}
    conf = np.zeros((12, 12), dtype=np.int64)
    for name in names:
        bidx = all_names.index(name)
        t0 = time.time()
        raw = load_building(cache[name])
        n = raw["xyz"].shape[0]
        logger.info(f"[{bidx + 1}/{len(all_names)}] {name}: {n} points (seed {args.seed + bidx})")
        final_idx, final_classes, final_scores, sem_full = process_building(
            model, raw, args.footprint, args.overlap, args.point_max, scale, min_spatial_shape,
            args.merge_iou, args.min_inst_points, logger, seed=args.seed + bidx)
        gt_sem = np.asarray(raw["sem"]).astype(np.int64)
        gt_inst = np.asarray(raw["inst"]).astype(np.int64)
        cov = sem_full >= 0
        conf += np.bincount(gt_sem[cov] * 12 + sem_full[cov], minlength=144).reshape(12, 12)
        sizes = np.array([len(x) for x in final_idx], dtype=np.int64)
        np.savez_compressed(
            osp.join(args.out_dir, f"{name}_pred.npz"),
            idx=np.concatenate(final_idx) if final_idx else np.zeros(0, np.int64),
            offsets=np.concatenate([[0], np.cumsum(sizes)]),
            classes=np.array(final_classes, dtype=np.int32),
            scores=np.array(final_scores, dtype=np.float32))
        t1 = time.time()
        gt_instances, pred_instances = evaluator.associate_instances(
            LazyMasks(final_idx, n), final_classes, final_scores, gt_sem, gt_inst)
        scenes.append(dict(gt=gt_instances, pred=pred_instances))
        one = evaluator.evaluate_matches([scenes[-1]])
        per_building[name] = dict(ap=float(one["all_ap"]), ap50=float(one["all_ap_50"]),
                                  ap25=float(one["all_ap_25"]), n_pred=len(final_idx),
                                  semantic_coverage=float(cov.mean()))
        logger.info(f"  building AP/AP50/AP25 {one['all_ap']:.4f}/{one['all_ap_50']:.4f}/"
                    f"{one['all_ap_25']:.4f}; associate {time.time() - t1:.1f}s; "
                    f"total {time.time() - t0:.1f}s")

    res = evaluator.evaluate_matches(scenes)
    iou = np.diag(conf) / np.maximum(conf.sum(0) + conf.sum(1) - np.diag(conf), 1)
    present = conf.sum(1) > 0
    out = dict(split=args.split, buildings=names, ckpt=args.ckpt,
               ap=float(res["all_ap"]), ap50=float(res["all_ap_50"]), ap25=float(res["all_ap_25"]),
               classes={k: {kk: float(vv) for kk, vv in v.items()} for k, v in res["classes"].items()},
               per_building=per_building,
               diag_semantic_miou_present_classes=float(iou[present].mean()),
               diag_semantic_iou={RAW_CLASS_NAMES[i]: float(iou[i]) for i in range(12) if present[i]},
               settings=dict(footprint=args.footprint, overlap=args.overlap,
                             point_max=args.point_max, merge_iou=args.merge_iou,
                             min_inst_points=args.min_inst_points, seed=args.seed))
    json.dump(out, open(osp.join(args.out_dir, "result.json"), "w"), indent=1)
    logger.info("=== Full-scene {} result: AP/AP50/AP25 {:.4f}/{:.4f}/{:.4f} ===".format(
        args.split, out["ap"], out["ap50"], out["ap25"]))
    for name_, c in out["classes"].items():
        logger.info(f"  {name_:12s} AP/AP50/AP25: {c['ap']:.4f}/{c['ap50']:.4f}/{c['ap25']:.4f}")
    logger.info(f"  [diagnostic, not protocol] full-res semantic mIoU (present classes) "
                f"{out['diag_semantic_miou_present_classes']:.4f}")


if __name__ == "__main__":
    main()
