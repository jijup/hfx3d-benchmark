"""
Standalone full-building evaluation for RandLA-Net on HFX3D.

Uses 200 crops per building (vs 20 during training) for stable coverage.
Loads checkpoint_best.tar and reports OA/mP/mR/mF1/mIoU + per-class.
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Override crop count before dataset module is fully initialised
import hfx3d_dataset
hfx3d_dataset.VAL_CROPS_PER_CLOUD = 200

from helper_hfx3d import ConfigHFX3D as cfg
from hfx3d_dataset import HFX3DDataset
from RandLANet_HFX3D import Network, compute_loss, compute_acc, IoUCalculator

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='/project/6118401/abdiaziz/data/HFX3D_Splits')
parser.add_argument('--checkpoint', default='output_hfx3d_v4/checkpoint_best.tar')
parser.add_argument('--batch_size', type=int, default=4)
FLAGS = parser.parse_args()


def log(msg):
    print(msg, flush=True)


# ── Dataset ────────────────────────────────────────────────────────────────
log(f"Loading val dataset from {FLAGS.data_root} (200 crops/building) ...")
test_ds = HFX3DDataset('validation', data_root=FLAGS.data_root)
test_loader = DataLoader(
    test_ds,
    batch_size=FLAGS.batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=test_ds.collate_fn,
    pin_memory=True,
)
log(f"Val batches: {len(test_loader)}")


# ── Model ──────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Network(cfg)
net.to(device)

if not os.path.isfile(FLAGS.checkpoint):
    raise FileNotFoundError(f"Checkpoint not found: {FLAGS.checkpoint}")

ckpt = torch.load(FLAGS.checkpoint, map_location=device)
net.load_state_dict(ckpt['model_state_dict'])
log(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

if torch.cuda.device_count() > 1:
    log(f"Using {torch.cuda.device_count()} GPUs")
    net = nn.DataParallel(net)


def move_to_device(batch):
    for key in batch:
        if isinstance(batch[key], list):
            batch[key] = [t.to(device) for t in batch[key]]
        else:
            batch[key] = batch[key].to(device)
    return batch


# ── Evaluation ─────────────────────────────────────────────────────────────
test_ds.reset_possibilities()
net.eval()
iou_calc = IoUCalculator(cfg)

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        batch = move_to_device(batch)
        end_pts = net(batch)
        _, end_pts = compute_loss(end_pts, cfg)
        _, end_pts = compute_acc(end_pts)
        iou_calc.add_data(end_pts)
        if (i + 1) % 50 == 0:
            log(f"  batch {i+1}/{len(test_loader)}")

mean_iou, mean_p, mean_r, mean_f1, oa, iou_list, p_list, r_list, f1_list = iou_calc.compute_iou()

log("")
log(f"Test OA/mP/mR/mF1/mIoU: {oa*100:.1f}% / {mean_p*100:.1f}% / {mean_r*100:.1f}% / {mean_f1*100:.1f}% / {mean_iou*100:.1f}%")
log("")
log("Per-class results:")
CLASS_NAMES = ['wall', 'window', 'door', 'balcony', 'vegetation', 'stairs',
               'terrain', 'roof', 'blinds', 'other', 'column', 'arch']
for c, name in enumerate(CLASS_NAMES):
    log(f"  {name:<12s}  IoU/P/R/F1 = {iou_list[c]*100:.1f}% / "
        f"{p_list[c]*100:.1f}% / {r_list[c]*100:.1f}% / {f1_list[c]*100:.1f}%")
