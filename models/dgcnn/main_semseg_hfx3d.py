#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DGCNN semantic segmentation on HFX3D urban facade dataset.
Adapted from AnTao97/dgcnn.pytorch (https://github.com/AnTao97/dgcnn.pytorch), main_semseg_s3dis.py
Features: xyz (3) + rgb (3) + intensity (1) = 7 channels, 12 classes
"""

from __future__ import print_function
import os
import time
import argparse
import numpy as np
import h5py
from scipy.spatial import cKDTree
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import Dataset, DataLoader
from model import get_graph_feature
from util import cal_loss, IOStream


CLASS_NAMES = ['wall', 'window', 'door', 'balcony', 'vegetation',
               'stairs', 'terrain', 'roof', 'blinds', 'other', 'column', 'arch']
NUM_CLASSES = 12


class HFX3D(Dataset):
    """
    Each __getitem__ returns num_points randomly sampled from one building.
    data_root must contain train/ and test/ subdirs with .h5 files.
    Data is already normalised by the preprocessor (xyz mean-centred metric, rgb [0,1], intensity global p1/p99 [0,1]).
    """
    def __init__(self, data_root, split='train', num_points=4096,
                 samples_per_building=50):
        self.num_points = num_points
        self.split = split

        split_dir = os.path.join(data_root, split)
        files = sorted([
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir) if f.endswith('.h5')
        ])

        self.points = []
        self.labels = []
        for fpath in files:
            with h5py.File(fpath, 'r') as f:
                xyz = f['point_cloud/xyz'][:].astype(np.float32)
                rgb = f['point_cloud/rgb'][:].astype(np.float32)   # already [0,1]
                inten = f['point_cloud/intensity'][:].astype(np.float32)
                lbl = f['labels/semantic'][:].astype(np.int64)

            if inten.ndim == 2:
                inten = inten[:, 0]
            inten = inten.reshape(-1, 1)

            # xyz already mean-centred (metric scale) by preprocessor
            feats = np.concatenate([xyz, rgb, inten], axis=1)
            self.points.append(feats)
            self.labels.append(lbl)

        # KD-tree per building for spatial (local) crop sampling — see __getitem__.
        self.trees = [cKDTree(pts[:, :3]) for pts in self.points]

        spb = samples_per_building if split == 'train' else 20
        self.index = [(bi, si) for bi in range(len(self.points)) for si in range(spb)]
        print('HFX3D %s: %d buildings, %d samples/epoch, %d pts/sample' % (
            split, len(files), len(self.index), num_points))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        bi, si = self.index[idx]
        pts = self.points[bi]
        lbl = self.labels[bi]
        tree = self.trees[bi]
        N = pts.shape[0]

        if self.split == 'train':
            rng = np.random.default_rng(None)
        else:
            rng = np.random.default_rng(bi * 1000 + si)

        # Spatial local crop: pick a random centre point, take its num_points
        # nearest neighbours. This matches the test-time KNN crops in
        # full_scene_infer.py, so the k-NN EdgeConv graph sees genuine local
        # geometry. The old code sampled num_points scattered across the whole
        # ~19M-point building (k-NN neighbours metres apart → no local structure,
        # and rare classes almost never sampled).
        k = min(self.num_points, N)
        center = pts[rng.integers(N), :3]
        _, nn = tree.query(center, k=k)
        nn = np.asarray(nn).reshape(-1)
        if k < self.num_points:
            pad = rng.choice(k, self.num_points - k, replace=True)
            nn = np.concatenate([nn, nn[pad]])
        return pts[nn], torch.LongTensor(lbl[nn])


class DGCNN_semseg_hfx3d(nn.Module):
    """DGCNN semseg: 7-channel input, 12 output classes."""
    def __init__(self, args):
        super().__init__()
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        # 7 features * 2 (self + neighbour diff) = 14
        self.conv1 = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=1, bias=False), self.bn1,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False), self.bn2,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False), self.bn3,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False), self.bn4,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False), self.bn5,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(
            nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False), self.bn6,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(
            nn.Conv1d(args.emb_dims + 192, 512, kernel_size=1, bias=False), self.bn7,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False), self.bn8,
            nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, NUM_CLASSES, kernel_size=1, bias=False)

    def forward(self, x):
        # x: (B, 7, N)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k)         # (B, 14, N, k)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.max(dim=-1, keepdim=False)[0]        # (B, 64, N)

        x = get_graph_feature(x1, k=self.k)         # (B, 128, N, k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=False)[0]        # (B, 64, N)

        x = get_graph_feature(x2, k=self.k)         # (B, 128, N, k)
        x = self.conv5(x)
        x3 = x.max(dim=-1, keepdim=False)[0]        # (B, 64, N)

        x = torch.cat((x1, x2, x3), dim=1)          # (B, 192, N)
        x = self.conv6(x)                            # (B, emb_dims, N)
        x_global = x.max(dim=-1, keepdim=True)[0].repeat(1, 1, num_points)
        x = torch.cat((x_global, x1, x2, x3), dim=1)  # (B, emb_dims+192, N)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)                            # (B, 12, N)
        return x


def calculate_sem_IoU(pred_np, seg_np):
    I_all = np.zeros(NUM_CLASSES)
    U_all = np.zeros(NUM_CLASSES)
    P_all = np.zeros(NUM_CLASSES)  # predicted count per class (TP+FP)
    T_all = np.zeros(NUM_CLASSES)  # GT count per class (TP+FN)
    total_correct = 0
    total_points = 0
    for i in range(seg_np.shape[0]):
        for c in range(NUM_CLASSES):
            I = np.sum((pred_np[i] == c) & (seg_np[i] == c))
            U = np.sum((pred_np[i] == c) | (seg_np[i] == c))
            I_all[c] += I
            U_all[c] += U
            P_all[c] += np.sum(pred_np[i] == c)
            T_all[c] += np.sum(seg_np[i] == c)
        total_correct += np.sum(pred_np[i] == seg_np[i])
        total_points += seg_np[i].shape[0]
    iou = np.where(U_all > 0, I_all / U_all, np.nan)
    precision = np.where(P_all > 0, I_all / P_all, np.nan)
    recall = np.where(T_all > 0, I_all / T_all, np.nan)
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall + 1e-9), np.nan)
    oa = total_correct / (total_points + 1e-9)
    return iou, precision, recall, f1, oa


def train(args, io):
    train_loader = DataLoader(
        HFX3D(args.data_root, split='train',
              num_points=args.num_points,
              samples_per_building=args.samples_per_building),
        num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)

    val_loader = DataLoader(
        HFX3D(args.data_root, split='val',
              num_points=args.num_points, samples_per_building=20),
        num_workers=4, batch_size=args.test_batch_size, shuffle=False)

    device = torch.device('cuda' if args.cuda else 'cpu')
    model = DGCNN_semseg_hfx3d(args).to(device)
    model = nn.DataParallel(model)
    io.cprint('GPUs: %d' % torch.cuda.device_count())

    if args.use_sgd:
        opt = optim.SGD(model.parameters(), lr=args.lr * 100,
                        momentum=args.momentum, weight_decay=1e-4)
    else:
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    else:
        scheduler = StepLR(opt, 20, 0.5, args.epochs)

    best_val_iou = 0
    for epoch in range(args.epochs):
        # ── train ──
        model.train()
        train_loss = count = 0.0
        train_true_seg, train_pred_seg = [], []

        for data, seg in train_loader:
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)   # (B,N,7) → (B,7,N)
            opt.zero_grad()
            seg_pred = model(data)         # (B,12,N)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = F.cross_entropy(seg_pred.view(-1, NUM_CLASSES), seg.view(-1))
            loss.backward()
            opt.step()
            count += data.size(0)
            train_loss += loss.item() * data.size(0)
            pred = seg_pred.max(dim=2)[1]
            train_true_seg.append(seg.cpu().numpy())
            train_pred_seg.append(pred.detach().cpu().numpy())

        if args.scheduler == 'cos':
            scheduler.step()
        else:
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()

        train_true_seg = np.concatenate(train_true_seg)
        train_pred_seg = np.concatenate(train_pred_seg)
        train_ious, train_prec, train_rec, train_f1, train_oa = calculate_sem_IoU(train_pred_seg, train_true_seg)
        io.cprint('Train %3d  loss=%.4f  OA/mP/mR/mF1/mIoU=%.2f%%/%.2f%%/%.2f%%/%.2f%%/%.2f%%' % (
            epoch, train_loss / count,
            train_oa * 100, np.nanmean(train_prec) * 100, np.nanmean(train_rec) * 100,
            np.nanmean(train_f1) * 100, np.nanmean(train_ious) * 100))

        # ── test ──
        model.eval()
        val_loss = count = 0.0
        val_true_seg, val_pred_seg = [], []

        with torch.no_grad():
            for data, seg in val_loader:
                data, seg = data.to(device), seg.to(device)
                data = data.permute(0, 2, 1)
                seg_pred = model(data)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                loss = F.cross_entropy(seg_pred.view(-1, NUM_CLASSES), seg.view(-1))
                count += data.size(0)
                val_loss += loss.item() * data.size(0)
                pred = seg_pred.max(dim=2)[1]
                val_true_seg.append(seg.cpu().numpy())
                val_pred_seg.append(pred.detach().cpu().numpy())

        val_true_seg = np.concatenate(val_true_seg)
        val_pred_seg = np.concatenate(val_pred_seg)
        val_ious, val_prec, val_rec, val_f1, val_oa = calculate_sem_IoU(val_pred_seg, val_true_seg)
        miou = np.nanmean(val_ious)
        io.cprint('Val   %3d  loss=%.4f  OA/mP/mR/mF1/mIoU=%.2f%%/%.2f%%/%.2f%%/%.2f%%/%.2f%%' % (
            epoch, val_loss / count,
            val_oa * 100, np.nanmean(val_prec) * 100, np.nanmean(val_rec) * 100,
            np.nanmean(val_f1) * 100, miou * 100))
        for c, name in enumerate(CLASS_NAMES):
            io.cprint('  %-12s  IoU/P/R/F1=%.1f%%/%.1f%%/%.1f%%/%.1f%%' % (
                name, np.nan_to_num(val_ious[c]) * 100, np.nan_to_num(val_prec[c]) * 100,
                np.nan_to_num(val_rec[c]) * 100, np.nan_to_num(val_f1[c]) * 100))

        if miou >= best_val_iou:
            best_val_iou = miou
            os.makedirs('outputs/%s/models' % args.exp_name, exist_ok=True)
            import shutil as _shl  # _DGCNN_SAVE_
            _dst = 'outputs/%s/models/best_model.t7' % args.exp_name
            _tmp = '/tmp/dgcnn_ckpt_{}.t7'.format(os.getpid())
            torch.save(model.state_dict(), _tmp)
            for _att in range(10):
                try:
                    _shl.copy2(_tmp, _dst); break
                except Exception as _e:
                    if _att < 9:
                        print(f'Copy attempt {_att+1}/10 failed: {_e}. Retrying in 10s...')
                        time.sleep(10)
                    else: os.remove(_tmp); raise
            os.remove(_tmp)
            io.cprint('*** Saved best model  mIoU=%.2f%%' % (best_val_iou * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='dgcnn_hfx3d')
    parser.add_argument('--data_root', type=str,
        default=os.environ.get("HFX3D_DATA_ROOT", "/project/6118401/abdiaziz/data/HFX3D_Splits"))
    parser.add_argument('--num_points', type=int, default=4096)
    parser.add_argument('--samples_per_building', type=int, default=100)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--emb_dims', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--scheduler', type=str, default='cos',
                        choices=['cos', 'step'])
    parser.add_argument('--use_sgd', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')  # _DGCNN_SEED_
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    os.makedirs('outputs/%s/models' % args.exp_name, exist_ok=True)
    io = IOStream('outputs/%s/run.log' % args.exp_name)
    io.cprint(str(args))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train(args, io)
