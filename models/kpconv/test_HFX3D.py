#
#      Full-scene test script for KPConv on HFX3D.
#      Adapted from HuguesTHOMAS/KPConv-PyTorch (https://github.com/HuguesTHOMAS/KPConv-PyTorch), test_models.py — uses the 'test' split.
#
#      Usage:
#        python test_HFX3D.py results/Log_YYYY-MM-DD_HH-MM-SS [-1]
#        (second arg: checkpoint index, -1 = last, default = current_chkp.tar)
#

import os
import sys
import builtins
import numpy as np
import torch

from datasets.HFX3D import HFX3DDataset, HFX3DSampler, HFX3DCollate
from torch.utils.data import DataLoader
from utils.config import Config
from utils.tester import ModelTester
import utils.tester as tester_module
from models.architectures import KPFCNN

# tester.py's cloud_segmentation_test() only ever prints per-class IoU/mIoU --
# it never computes OA/precision/recall/F1. Capture the exact confusion
# matrix it builds internally (by wrapping the function it calls it with)
# so we can derive those metrics ourselves without touching the shared file.
#
# IoU_from_confusions(C) is called for BOTH the "sub clouds" confusion (an
# approximate, rescaled matrix from subsampled crops) and the "full clouds"
# confusion (the real reprojected full-resolution matrix, printed as the
# authoritative mIoU) -- from two different call sites in tester.py, using
# the same function. A naive "last call wins" capture is wrong: tester.py's
# vote loop always does one more unpaired "sub clouds" check right before it
# exits at num_votes, so the truly-last captured confusion is a sub-cloud
# one, not the full-cloud one we want. Instead, watch for the literal
# 'Confusion on full clouds' print that always immediately precedes the
# full-cloud IoU_from_confusions call, and only capture that one.
_captured_full_confusions = []
_pending_full = [False]

_orig_builtin_print = builtins.print
def _tracking_print(*args, **kwargs):
    if args and args[0] == 'Confusion on full clouds':
        _pending_full[0] = True
    return _orig_builtin_print(*args, **kwargs)
builtins.print = _tracking_print

_orig_iou_from_confusions = tester_module.IoU_from_confusions
def _capturing_iou_from_confusions(C):
    if _pending_full[0]:
        _captured_full_confusions.append(C.copy())
        _pending_full[0] = False
    return _orig_iou_from_confusions(C)
tester_module.IoU_from_confusions = _capturing_iou_from_confusions

# tester.py's cloud-export step (predictions/probs/potentials .ply) has hit
# transient Lustre write failures (OSError "N requested and M written") and,
# separately, is multi-GB per run with no analytical payoff -- these exports
# aren't needed for our metrics, so skip the write entirely (see below).
_orig_write_ply = tester_module.write_ply
_KP_PRED_DIR = os.environ.get('KP_PRED_DIR', '/home/a/abdiaziz/KPConv/test_preds_npy')
def _safe_write_ply(*args, **kwargs):
    # Intercept the reprojected full-cloud predictions and save them as .npy,
    # BEFORE the (Lustre-flaky) .ply write, so we always capture per-point preds.
    try:
        fname  = args[0] if len(args) > 0 else ''
        data   = args[1] if len(args) > 1 else []
        fields = args[2] if len(args) > 2 else []
        if 'preds' in fields:
            os.makedirs(_KP_PRED_DIR, exist_ok=True)
            preds = np.asarray(data[-1]).ravel().astype(np.uint8)
            stem = os.path.basename(str(fname))
            if stem.endswith('.ply'):
                stem = stem[:-4]
            if stem.endswith('.h5'):
                stem = stem[:-3]
            np.save(os.path.join(_KP_PRED_DIR, stem + '.npy'), preds)
            print('SAVED per-point preds -> {}.npy ({} pts)'.format(stem, preds.size))
    except Exception as e:
        print('WARNING: pred npy save failed: {}'.format(e))
    # Skip the actual .ply write entirely -- we only ever use the .npy
    # capture above plus the printed mIoU/confusion; the .ply exports are
    # multi-GB per run (probs+predictions+potentials) and have repeatedly
    # blown the 25GiB home quota with no analytical payoff.
tester_module.write_ply = _safe_write_ply


def main():
    if len(sys.argv) < 2:
        print('Usage: python test_HFX3D.py <results/Log_...> [chkp_idx]')
        sys.exit(1)

    chosen_log = sys.argv[1]
    chkp_idx = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if not os.path.exists(chosen_log):
        raise ValueError('Log does not exist: ' + chosen_log)

    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = sorted([f for f in os.listdir(chkp_path) if f.startswith('chkp')])

    if chkp_idx is None:
        chosen_chkp = os.path.join(chkp_path, 'current_chkp.tar')
    else:
        chosen_chkp = os.path.join(chkp_path, chkps[chkp_idx])

    print('Using checkpoint: ' + chosen_chkp)

    GPU_ID = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    config = Config()
    config.load(chosen_log)

    # feature_mode disambiguates 4-channel combos sharing the same
    # in_features_dim (see train_HFX3D.py) -- it's not part of KPConv's own
    # config save/load, so it's persisted as a sidecar file instead.
    feature_mode_file = os.path.join(chosen_log, 'feature_mode.txt')
    if os.path.exists(feature_mode_file):
        with open(feature_mode_file) as f:
            config.feature_mode = f.read().strip()
        print('Loaded feature_mode override: ' + config.feature_mode)

    # Increase validation size so all test buildings get full coverage
    config.validation_size = 200
    config.input_threads = 4

    print()
    print('Data Preparation')
    print('****************')

    # Use 'test' split for final numbers; switch to 'validation' for checkpoint selection
    test_dataset = HFX3DDataset(config, set='test', use_potentials=True)

    # KPConv's tester.py only computes/prints the confusion-matrix mIoU when
    # dataset.set == 'validation' (it assumes 'test' means withheld/unlabeled,
    # as in public benchmarks). Our test/ split has real ground-truth labels
    # (HFX3DDataset already loads validation_labels for set in ('validation',
    # 'test')), so flip the flag post-construction to get tester.py to report
    # real numbers. File discovery above already used the true test/ directory.
    test_dataset.set = 'validation'

    test_sampler = HFX3DSampler(test_dataset)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=test_sampler,
        collate_fn=HFX3DCollate,
        num_workers=config.input_threads,
        pin_memory=True,
    )

    test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    import time
    t1 = time.time()
    net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)

    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')

    # num_votes default is 100; at the observed convergence rate that would
    # take many hours (loop only breaks when last_min > num_votes). The full
    # reprojected confusion (what we need) is computed once per 10 "votes" of
    # coverage, so num_votes=10 gets us exactly that and then exits promptly.
    tester.cloud_segmentation_test(net, test_loader, config, num_votes=10)

    print('\n========== DERIVED METRICS (from final full-cloud confusion matrix) ==========')
    if _captured_full_confusions:
        # Last full-cloud confusion actually tagged as such (see capture
        # logic above) -- the reprojected, full-resolution matrix used for
        # the printed mIoU.
        C = _captured_full_confusions[-1].astype(np.float64)
        OA = 100 * np.trace(C) / (np.sum(C) + 1e-6)
        TP = np.diag(C)
        FP = np.sum(C, axis=0) - TP
        FN = np.sum(C, axis=1) - TP
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        names = [test_dataset.label_to_names[l] for l in test_dataset.label_values
                 if l not in test_dataset.ignored_labels]

        print('OA={:.2f}%  mP={:.2f}%  mR={:.2f}%  mF1={:.2f}%'.format(
            OA, 100 * np.mean(precision), 100 * np.mean(recall), 100 * np.mean(f1)))
        for n, p, r, f in zip(names, precision, recall, f1):
            print('  {:12s} P={:6.2f}%  R={:6.2f}%  F1={:6.2f}%'.format(n, 100 * p, 100 * r, 100 * f))
    else:
        print('WARNING: no confusion matrix captured -- derived metrics unavailable.')


if __name__ == '__main__':
    main()
