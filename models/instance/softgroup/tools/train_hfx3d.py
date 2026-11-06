"""Iteration-exact SoftGroup trainer for HFX3D.

Replaces tools/train.py's epoch loop for HFX3D runs:
* the budget is counted in OPTIMIZER STEPS on a GLOBAL counter shared by both
  stages (stage 1 = semantic/offset backbone, stage 2 = full model);
* each stage config has a `schedule` block: global_start, iters, lr,
  hold_iters (constant-LR steps) then cosine to min_lr (SoftGroup's own
  cosine_lr_after_step, in steps instead of epochs);
* a checkpoint is written at every global step that is a multiple of
  schedule.save_every (1800) and at the end of the stage;
* seed 42 for python/numpy/torch; the data stream is a pure function of
  (seed, global sample index) -- see softgroup/data/hfx3d.py -- so resuming
  from step s replays exactly the crops a non-interrupted run would see;
* SoftGroup's own per-block validate() is NOT run (protocol: reported numbers
  come only from tools/test_fullscene_softgroup_hfx3d.py).

Usage (inside the container, from the repo root):
  python tools/train_hfx3d.py configs/softgroup/softgroup_hfx3d_fresh_backbone.yaml \
      --work_dir work_dirs/hfx3d_fresh [--resume <ckpt>] [--stop_after N]
"""
import argparse
import datetime
import os
import os.path as osp
import random
import shutil
import time
from math import cos, pi

import numpy as np
import torch
import yaml
from munch import Munch
from torch.utils.data import DataLoader

from softgroup.data.hfx3d import HFX3DDataset
from softgroup.model import SoftGroup
from softgroup.util import build_optimizer, get_root_logger, load_checkpoint


def get_args():
    p = argparse.ArgumentParser('SoftGroup-HFX3D')
    p.add_argument('config')
    p.add_argument('--work_dir', required=True)
    p.add_argument('--resume', default=None, help='checkpoint of THIS stage to resume from')
    p.add_argument('--stop_after', type=int, default=0,
                   help='smoke test only: stop after N steps of this stage (schedule unchanged)')
    p.add_argument('--save_every', type=int, default=0, help='override schedule.save_every')
    p.add_argument('--num_workers', type=int, default=16)
    p.add_argument('--pretrain', default=None, help='override cfg.pretrain (stage-1 checkpoint)')
    return p.parse_args()


def lr_at(t, sch):
    """t = step index within the stage (0-based)."""
    if t < sch.hold_iters:
        return sch.lr
    span = max(sch.iters - sch.hold_iters, 1)
    return sch.min_lr + 0.5 * (sch.lr - sch.min_lr) * (1 + cos(pi * (t - sch.hold_iters) / span))


def worker_init_fn(worker_id):
    # Crops are seeded per sample (independent of workers); this only makes any
    # stray global-RNG use inside workers deterministic too.
    s = 42 + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def save_ckpt(path, model, optimizer, global_step, stage, work_dir):
    state = {
        'net': {k: v.cpu() for k, v in model.state_dict().items()},
        'optimizer': optimizer.state_dict(),
        'global_step': global_step,
        'stage': stage,
        'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()},
    }
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)
    latest = osp.join(work_dir, f'latest_{stage}.pth')
    if osp.lexists(latest):
        os.remove(latest)
    os.symlink(osp.basename(path), latest)


def main():
    args = get_args()
    cfg_txt = open(args.config).read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    sch = cfg.schedule
    if args.pretrain:
        cfg.pretrain = args.pretrain
    seed = cfg.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.makedirs(args.work_dir, exist_ok=True)
    stage = sch.stage
    ts = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger = get_root_logger(log_file=osp.join(args.work_dir, f'{stage}_{ts}.log'))
    logger.info(f'Config:\n{cfg_txt}')
    shutil.copy(args.config, osp.join(args.work_dir, osp.basename(args.config)))

    model = SoftGroup(**cfg.model).cuda()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'trainable params: {n_params}')
    optimizer = build_optimizer(model, cfg.optimizer)

    global_step = sch.global_start
    end_step = sch.global_start + sch.iters
    if args.resume:
        logger.info(f'Resume from {args.resume}')
        load_checkpoint(args.resume, logger, model, optimizer=optimizer)
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        assert ck['stage'] == stage, f"resume checkpoint is from stage {ck['stage']}"
        global_step = ck['global_step']
        random.setstate(ck['rng']['python'])
        np.random.set_state(ck['rng']['numpy'])
        torch.set_rng_state(ck['rng']['torch'])
        torch.cuda.set_rng_state_all(ck['rng']['cuda'])
    elif cfg.get('pretrain'):
        logger.info(f'Load pretrain (model weights only) from {cfg.pretrain}')
        load_checkpoint(cfg.pretrain, logger, model)
    if global_step >= end_step:
        logger.info(f'stage {stage} already complete at step {global_step}')
        return

    bs = cfg.dataloader.train.batch_size
    dataset = HFX3DDataset(**cfg.data.train, logger=logger, seed=seed)
    dataset.set_range(global_step * bs, (end_step - global_step) * bs)
    loader = DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=True,
                        num_workers=args.num_workers, collate_fn=dataset.collate_fn,
                        pin_memory=True, worker_init_fn=worker_init_fn,
                        prefetch_factor=4 if args.num_workers > 0 else None,
                        persistent_workers=False)
    save_every = args.save_every or sch.save_every
    logger.info(f'stage={stage} global steps {global_step} -> {end_step} '
                f'(batch {bs}, lr {sch.lr}, hold {sch.hold_iters}, save_every {save_every})')

    model.train()
    stage_start = global_step
    t_data, t_iter, end = 0.0, 0.0, time.time()
    win = []  # recent iteration times
    for batch in loader:
        td = time.time() - end
        t = global_step - sch.global_start
        lr = lr_at(t, sch)
        for g in optimizer.param_groups:
            g['lr'] = lr
        loss, log_vars = model(batch, return_loss=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        ti = time.time() - end
        end = time.time()
        win.append(ti)
        win = win[-50:]
        if global_step % 10 == 0 or global_step - stage_start <= 5 or args.stop_after:
            eta = datetime.timedelta(seconds=int(np.mean(win) * (end_step - global_step)))
            mem = torch.cuda.max_memory_allocated() / 2**30
            npts = int(batch['coords'].shape[0])
            fp = float(batch['coords_float'].double().abs().sum())  # data-stream fingerprint
            s = (f'[{stage}] step {global_step}/{end_step} lr {lr:.2e} pts {npts} fp {fp:.1f} '
                 f'data {td:.2f}s iter {ti:.2f}s (avg50 {np.mean(win):.2f}s) eta {eta} '
                 f'maxmem {mem:.1f}G')
            for k, v in log_vars.items():
                s += f' {k} {v:.4f}'
            logger.info(s)
        if not np.isfinite(log_vars['loss']):
            logger.error(f'non-finite loss at step {global_step}; aborting')
            raise SystemExit(1)
        stop = args.stop_after and global_step - stage_start >= args.stop_after
        if global_step % save_every == 0 or global_step == end_step or stop:
            path = osp.join(args.work_dir, f'step_{global_step:05d}.pth')
            save_ckpt(path, model, optimizer, global_step, stage, args.work_dir)
            logger.info(f'saved {path}')
        if stop:
            logger.info(f'--stop_after {args.stop_after} reached at step {global_step}')
            break
    logger.info(f'stage {stage} done at global step {global_step}; '
                f'max GPU mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G')


if __name__ == '__main__':
    main()
