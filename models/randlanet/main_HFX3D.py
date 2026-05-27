"""
RandLA-Net training / evaluation entry point for HFX3D.

Usage
-----
python main_HFX3D.py [--checkpoint_path <path>] [--log_dir <dir>]
                     [--max_epoch 100] [--batch_size 4]
                     [--data_root /path/to/HFX3D_Splits]
                     [--gpus 2]
"""

import os, sys, argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Local imports (all from ~/RandLA-Net/)
from helper_hfx3d import ConfigHFX3D as cfg, DataProcessing as DP
from hfx3d_dataset import HFX3DDataset, NUM_CLASSES
from RandLANet_HFX3D import Network, compute_loss, compute_acc, IoUCalculator


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default=None)
parser.add_argument('--log_dir', default='output_hfx3d')
parser.add_argument('--max_epoch', type=int, default=cfg.max_epoch)
parser.add_argument('--batch_size', type=int, default=cfg.batch_size)
parser.add_argument('--data_root', default='/home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits')
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--seed', type=int, default=42, help='Random seed')  # _RANDLA_SEED_
FLAGS = parser.parse_args()
np.random.seed(FLAGS.seed)
torch.manual_seed(FLAGS.seed)

cfg.max_epoch = FLAGS.max_epoch
cfg.batch_size = FLAGS.batch_size

LOG_DIR = FLAGS.log_dir
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')


def log_string(s):
    LOG_FOUT.write(s + '\n')
    LOG_FOUT.flush()
    print(s)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


log_string(f"Loading datasets from {FLAGS.data_root} ...")
TRAIN_DATASET = HFX3DDataset('training',  data_root=FLAGS.data_root)
TEST_DATASET  = HFX3DDataset('val', data_root=FLAGS.data_root)
log_string(f"Train clouds: {len(TRAIN_DATASET)}  Val clouds: {len(TEST_DATASET)}")

TRAIN_LOADER = DataLoader(
    TRAIN_DATASET, batch_size=FLAGS.batch_size, shuffle=True,
    num_workers=0, worker_init_fn=worker_init_fn,
    collate_fn=TRAIN_DATASET.collate_fn, pin_memory=True)

TEST_LOADER = DataLoader(
    TEST_DATASET, batch_size=FLAGS.batch_size, shuffle=False,
    num_workers=0, worker_init_fn=worker_init_fn,
    collate_fn=TEST_DATASET.collate_fn, pin_memory=True)

log_string(f"Train batches: {len(TRAIN_LOADER)}  Val batches: {len(TEST_LOADER)}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = Network(cfg)
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)

start_epoch = 0
if FLAGS.checkpoint_path is not None and os.path.isfile(FLAGS.checkpoint_path):
    ckpt = torch.load(FLAGS.checkpoint_path, map_location=device)
    net.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch']
    log_string(f"Loaded checkpoint {FLAGS.checkpoint_path} (epoch {start_epoch})")

if torch.cuda.device_count() > 1:
    log_string(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
    net = nn.DataParallel(net)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def adjust_lr(optimizer, epoch):
    lr = optimizer.param_groups[0]['lr']
    lr = lr * cfg.lr_decays.get(epoch, 1.0)
    for g in optimizer.param_groups:
        g['lr'] = lr


def move_to_device(batch):
    for key in batch:
        if isinstance(batch[key], list):
            batch[key] = [t.to(device) for t in batch[key]]
        else:
            batch[key] = batch[key].to(device)
    return batch


def train_one_epoch(epoch):
    adjust_lr(optimizer, epoch)
    net.train()
    iou_calc = IoUCalculator(cfg)
    stat = {}

    for i, batch in enumerate(TRAIN_LOADER):
        batch = move_to_device(batch)
        optimizer.zero_grad()
        end_pts = net(batch)
        loss, end_pts = compute_loss(end_pts, cfg)
        loss.backward()
        optimizer.step()
        acc, end_pts = compute_acc(end_pts)
        iou_calc.add_data(end_pts)

        for k in end_pts:
            if any(x in k for x in ('loss', 'acc', 'iou')):
                stat[k] = stat.get(k, 0) + end_pts[k].item()

        if (i + 1) % 50 == 0:
            log_string(f'  batch {i+1:04d}')
            for k in sorted(stat.keys()):
                log_string(f'    {k}: {stat[k]/50:.4f}')
                stat[k] = 0

    mean_iou, mean_precision, mean_recall, mean_f1, oa, iou_list, prec_list, rec_list, f1_list = iou_calc.compute_iou()
    log_string(f'Train OA/mP/mR/mF1/mIoU: {oa*100:.1f}% / {mean_precision*100:.1f}% / {mean_recall*100:.1f}% / {mean_f1*100:.1f}% / {mean_iou*100:.1f}%')
    iou_str = '  '.join(f'{v*100:.1f}' for v in iou_list)
    log_string(f'Per-class IoU: {iou_str}')


def eval_one_epoch():
    TEST_DATASET.reset_possibilities()
    net.eval()
    iou_calc = IoUCalculator(cfg)
    stat = {}

    with torch.no_grad():
        for i, batch in enumerate(TEST_LOADER):
            batch = move_to_device(batch)
            end_pts = net(batch)
            loss, end_pts = compute_loss(end_pts, cfg)
            acc, end_pts = compute_acc(end_pts)
            iou_calc.add_data(end_pts)

            for k in end_pts:
                if any(x in k for x in ('loss', 'acc', 'iou')):
                    stat[k] = stat.get(k, 0) + end_pts[k].item()

    n = len(TEST_LOADER)
    for k in sorted(stat.keys()):
        log_string(f'  val {k}: {stat[k]/n:.4f}')

    mean_iou, mean_precision, mean_recall, mean_f1, oa, iou_list, prec_list, rec_list, f1_list = iou_calc.compute_iou()
    log_string(f'Val OA/mP/mR/mF1/mIoU: {oa*100:.1f}% / {mean_precision*100:.1f}% / {mean_recall*100:.1f}% / {mean_f1*100:.1f}% / {mean_iou*100:.1f}%')
    iou_str = '  '.join(f'{v*100:.1f}' for v in iou_list)
    log_string(f'Per-class IoU: {iou_str}')
    return mean_iou


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def train(start_epoch):
    best_miou = 0.0
    for epoch in range(start_epoch, cfg.max_epoch):
        log_string(f'\n**** EPOCH {epoch:03d}  {datetime.now()} ****')
        train_one_epoch(epoch)

        if epoch == 0 or (epoch + 1) % 5 == 0:
            log_string(f'-- Eval epoch {epoch} --')
            miou = eval_one_epoch()
            if miou > best_miou:
                best_miou = miou
                save_path = os.path.join(LOG_DIR, 'checkpoint_best.tar')
                core = net.module if hasattr(net, 'module') else net
                _tmp = '/tmp/randla_best_{}.tar'.format(os.getpid())  # _RANDLA_BEST_SAVE_
                torch.save({'epoch': epoch+1,
                            'model_state_dict': core.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'miou': best_miou}, _tmp)
                import shutil as _shl
                for _att in range(10):
                    try:
                        _shl.copy2(_tmp, save_path); break
                    except Exception as _e:
                        if _att < 9:
                            log_string(f'Copy attempt {_att+1}/10 failed: {_e}. Retrying...')
                            time.sleep(10)
                        else: os.remove(_tmp); raise
                os.remove(_tmp)
                log_string(f'  Saved best model (mIoU={best_miou*100:.1f}%) -> {save_path}')

        # Always save latest checkpoint
        save_path = os.path.join(LOG_DIR, 'checkpoint.tar')
        core = net.module if hasattr(net, 'module') else net
        _tmp = '/tmp/randla_latest_{}.tar'.format(os.getpid())  # _RANDLA_LATEST_SAVE_
        torch.save({'epoch': epoch+1,
                    'model_state_dict': core.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, _tmp)
        import shutil as _shl
        for _att in range(10):
            try:
                _shl.copy2(_tmp, save_path); break
            except Exception as _e:
                if _att < 9:
                    log_string(f'Copy attempt {_att+1}/10 failed: {_e}. Retrying...')
                    time.sleep(10)
                else: os.remove(_tmp); raise
        os.remove(_tmp)


if __name__ == '__main__':
    train(start_epoch)
