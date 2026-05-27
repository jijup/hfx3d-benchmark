"""
Full-scene test evaluation for DGCNN on HFX3D.

Uses possibility-based KNN inference (full_scene_infer.FullSceneEvaluator) to
guarantee 100% point coverage per building before computing metrics.
No per-crop xyz centering — matches DGCNN training data loader exactly.

Input features: xyz (3) + rgb (3) + intensity (1) = 7 channels.
Model output:   (B, 12, N) raw logits  →  softmax via torch.softmax(dim=1).
"""

import argparse
import glob
import h5py
import importlib.util
import logging
import os
import sys
import time
import types
import numpy as np
import torch
from pathlib import Path

from full_scene_infer import FullSceneEvaluator, compute_metrics

NUM_CLASSES = 12
CLASS_NAMES = [
    "wall", "window", "door", "balcony", "vegetation", "stairs",
    "terrain", "roof", "blinds", "other", "column", "arch",
]


def parse_args():
    parser = argparse.ArgumentParser("DGCNN HFX3D Full-Scene Test")
    parser.add_argument("--exp_name", type=str, default="dgcnn_hfx3d")
    parser.add_argument("--data_root", type=str,
                        default="/home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--emb_dims", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--crop_size", type=int, default=4096,
                        help="Points per crop [default: 4096]")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def main(args):
    log_path = Path(f"./outputs/{args.exp_name}/test_fullscene.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("Test")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(log_path))
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(ch)

    def log(s):
        logger.info(s)

    # ── Load model ────────────────────────────────────────────────────────
    spec = importlib.util.spec_from_file_location(
        "main_semseg_hfx3d",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_semseg_hfx3d.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    DGCNN_semseg_hfx3d = mod.DGCNN_semseg_hfx3d

    model_args = types.SimpleNamespace(k=args.k, emb_dims=args.emb_dims, dropout=args.dropout)
    model = DGCNN_semseg_hfx3d(model_args).cuda()

    ckpt_path = f"./outputs/{args.exp_name}/models/best_model.t7"
    state = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    log(f"Loaded {ckpt_path}")
    log(f"k={args.k}  emb_dims={args.emb_dims}  crop_size={args.crop_size}  batch_size={args.batch_size}")

    # ── Run per-building inference ────────────────────────────────────────
    test_dir = os.path.join(args.data_root, "test")
    h5_files = sorted(glob.glob(os.path.join(test_dir, "*.h5")))
    log(f"Test buildings: {len(h5_files)}")

    all_tp = np.zeros(NUM_CLASSES, dtype=np.int64)
    all_fp = np.zeros(NUM_CLASSES, dtype=np.int64)
    all_fn = np.zeros(NUM_CLASSES, dtype=np.int64)

    for fp in h5_files:
        bname = os.path.basename(fp)
        with h5py.File(fp, "r") as f:
            xyz       = f["point_cloud/xyz"][:]
            rgb       = f["point_cloud/rgb"][:]
            intensity = f["point_cloud/intensity"][:]
            labels    = f["labels/semantic"][:].astype(np.int64)

        if intensity.ndim == 2:
            intensity = intensity[:, 0]
        feats = np.concatenate(
            [xyz, rgb, intensity.reshape(-1, 1)], axis=1
        ).astype(np.float32)
        N = feats.shape[0]
        log(f"\n--- {bname}: {N:,} pts ---")

        t0 = time.time()
        ev = FullSceneEvaluator(xyz, num_classes=NUM_CLASSES, crop_size=args.crop_size)
        log(f"  KDTree built in {time.time()-t0:.1f}s")

        batch_count = 0
        t1 = time.time()
        while not ev.is_done():
            pending_sel = []
            pending_crops = []

            for _ in range(args.batch_size):
                sel_idx, k_real, center_idx = ev.next_crop()
                crop = feats[sel_idx]          # (crop_size, 7) — no centering
                pending_sel.append((sel_idx, k_real, center_idx))
                pending_crops.append(crop)

            batch = (
                torch.from_numpy(np.stack(pending_crops))
                .float()
                .permute(0, 2, 1)  # (B, 7, crop_size)
                .cuda()
            )
            with torch.no_grad():
                logits = model(batch)          # (B, 12, crop_size) raw logits
                probs = torch.softmax(logits, dim=1).permute(0, 2, 1).cpu().numpy()  # (B, crop_size, 12)

            for i, (sidx, kr, ci) in enumerate(pending_sel):
                ev.accumulate(ci, sidx, probs[i], kr)

            batch_count += 1
            if batch_count % 100 == 0:
                cov = ev.coverage_report()
                elapsed = time.time() - t1
                log(f"  batch {batch_count}  coverage={cov['coverage_pct']:.1f}%  "
                    f"elapsed={elapsed:.0f}s")

        cov = ev.coverage_report()
        assert cov["coverage_pct"] == 100.0, (
            f"Coverage check failed: {cov['covered_points']}/{cov['total_points']} points covered"
        )
        log(f"  Coverage: {cov['coverage_pct']:.1f}%  "
            f"min/mean/max votes: {cov['min_votes']}/{cov['mean_votes']:.1f}/{cov['max_votes']}  "
            f"total batches: {batch_count}  time: {time.time()-t0:.0f}s")

        pred = ev.predictions()
        m = compute_metrics(pred, labels, NUM_CLASSES)
        log(f"  Building mIoU={m['mIoU']*100:.2f}%  OA={m['OA']*100:.2f}%  "
            f"mP={m['mP']*100:.2f}%  mR={m['mR']*100:.2f}%  mF1={m['mF1']*100:.2f}%")

        all_tp += m["tp"]
        all_fp += m["fp"]
        all_fn += m["fn"]

    # ── Aggregate across all buildings ───────────────────────────────────
    denom_iou = (all_tp + all_fp + all_fn).astype(np.float64)
    iou  = np.where(denom_iou > 0, all_tp / (denom_iou + 1e-10), 0.0)
    prec = np.where(all_tp + all_fp > 0,
                    all_tp / (all_tp + all_fp + 1e-10), 0.0)
    rec  = np.where(all_tp + all_fn > 0,
                    all_tp / (all_tp + all_fn + 1e-10), 0.0)
    f1   = np.where(prec + rec > 0,
                    2.0 * prec * rec / (prec + rec + 1e-10), 0.0)
    oa   = float(all_tp.sum()) / float((all_tp + all_fn).sum() + 1e-9)

    log("\n========== FINAL TEST RESULTS (all buildings combined) ==========")
    log(f"OA:    {oa*100:.2f}%")
    log(f"mIoU:  {iou.mean()*100:.2f}%")
    log(f"mP:    {prec.mean()*100:.2f}%")
    log(f"mR:    {rec.mean()*100:.2f}%")
    log(f"mF1:   {f1.mean()*100:.2f}%")
    log("\nPer-class  IoU / P / R / F1:")
    for c in range(NUM_CLASSES):
        log(f"  {CLASS_NAMES[c]:<12s}  "
            f"IoU:{iou[c]*100:6.2f}%  "
            f"P:{prec[c]*100:6.2f}%  "
            f"R:{rec[c]*100:6.2f}%  "
            f"F1:{f1[c]*100:6.2f}%")


if __name__ == "__main__":
    args = parse_args()
    main(args)
