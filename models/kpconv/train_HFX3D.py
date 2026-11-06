#
#      Training script for KPConv on HFX3D dataset.
#      Adapted from HuguesTHOMAS/KPConv-PyTorch (https://github.com/HuguesTHOMAS/KPConv-PyTorch), train_SensatUrban.py.
#
#      Phase 1: unweighted CE loss, seed 42.
#      For Phase 2: set segloss_balance='class' and class_w below.
#

import signal
import os
import sys
import numpy as np
import torch

from datasets.HFX3D import HFX3DDataset, HFX3DSampler, HFX3DCollate
from torch.utils.data import DataLoader
from utils.config import Config
from utils.trainer import ModelTrainer
from models.architectures import KPFCNN


class HFX3DConfig(Config):

    ####################
    # Dataset parameters
    ####################

    dataset = 'HFX3D'
    num_classes = None       # filled by dataset __init__
    dataset_task = 'cloud_segmentation'
    input_threads = 8        # DataLoader workers (use 0 if OOM from forking)

    #########################
    # Architecture definition
    #########################

    architecture = [
        'simple',
        'resnetb',
        'resnetb_strided',
        'resnetb',
        'resnetb',
        'resnetb_strided',
        'resnetb',
        'resnetb',
        'resnetb_strided',
        'resnetb_deformable',
        'resnetb_deformable',
        'resnetb_deformable_strided',
        'resnetb_deformable',
        'resnetb_deformable',
        'nearest_upsample',
        'unary',
        'nearest_upsample',
        'unary',
        'nearest_upsample',
        'unary',
        'nearest_upsample',
        'unary',
    ]

    ###################
    # KPConv parameters
    ###################

    num_kernel_points = 15

    # Sphere crop radius — 4m captures a meaningful facade patch
    in_radius = 4.0

    # Grid voxel size — 5cm suits dense facade LiDAR
    first_subsampling_dl = 0.05

    conv_radius = 2.5
    deform_radius = 6.0
    KP_extent = 1.2
    KP_influence = 'linear'
    aggregation_mode = 'sum'

    # Features: 1 (ones) + 3 (rgb) + 1 (intensity) = 5
    first_features_dim = 128
    in_features_dim = 7

    # Disambiguates 4-channel feature combos that share the same
    # in_features_dim (e.g. "1+rgb" vs "xyz+intensity" are both 4 channels).
    # None = legacy default (1+rgb) when in_features_dim == 4.
    feature_mode = None

    modulated = False
    use_batch_norm = True
    batch_norm_momentum = 0.02

    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0
    deform_lr_factor = 0.1
    repulse_extent = 1.2

    #####################
    # Training parameters
    #####################

    max_epoch = 200
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1 ** (1 / 100) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    batch_num = 6
    epoch_steps = 500
    validation_size = 50
    checkpoint_gap = 50

    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_noise = 0.001
    augment_color = 0.8

    # Phase 1: no class weighting
    segloss_balance = 'none'
    class_w = []

    # Phase 2 weights (uncomment to use):
    # segloss_balance = 'class'
    # class_w = [0.033, 0.151, 2.800, 1.000, 0.141, 1.273, 0.051, 1.400, 0.483, 0.519, 1.000, 5.000]

    saving = True
    saving_path = None


if __name__ == '__main__':

    # Reproducibility
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    GPU_ID = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: %s' % device)

    ###############
    # Previous chkp
    ###############

    previous_training_path = None
    chkp_idx = None

    if previous_training_path:
        chkp_path = os.path.join('results', previous_training_path, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']
        chosen_chkp = 'current_chkp.tar' if chkp_idx is None else np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join('results', previous_training_path, 'checkpoints', chosen_chkp)
    else:
        chosen_chkp = None

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    config = HFX3DConfig()
    if previous_training_path:
        config.load(os.path.join('results', previous_training_path))
        config.saving_path = None

    if len(sys.argv) > 1:
        config.saving_path = sys.argv[1]
    if len(sys.argv) > 2:
        config.in_features_dim = int(sys.argv[2])
    if len(sys.argv) > 3:
        config.feature_mode = sys.argv[3]

    if config.feature_mode:
        # KPConv's own Config.save()/load() only round-trip a fixed field
        # list (see utils/config.py) -- feature_mode isn't in it, so it
        # would silently vanish at test time via config.load(). Persist it
        # as a sidecar file tied to this results dir instead; test_HFX3D.py
        # reads it back explicitly.
        os.makedirs(config.saving_path, exist_ok=True)
        with open(os.path.join(config.saving_path, 'feature_mode.txt'), 'w') as f:
            f.write(config.feature_mode)

    training_dataset = HFX3DDataset(config, set='training', use_potentials=True)
    val_dataset = HFX3DDataset(config, set='validation', use_potentials=True)

    training_sampler = HFX3DSampler(training_dataset)
    val_sampler = HFX3DSampler(val_dataset)

    training_loader = DataLoader(
        training_dataset,
        batch_size=1,
        sampler=training_sampler,
        collate_fn=HFX3DCollate,
        num_workers=config.input_threads,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        sampler=val_sampler,
        collate_fn=HFX3DCollate,
        num_workers=config.input_threads,
        pin_memory=True,
    )

    # Calibrate batch size and neighborhood limits
    training_sampler.calibration(training_loader, verbose=True)
    val_sampler.calibration(val_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    import time
    t1 = time.time()
    net = KPFCNN(config, training_dataset.label_values, training_dataset.ignored_labels)

    print(net)
    print('Model parameters: {:d}'.format(
        sum(p.numel() for p in net.parameters() if p.requires_grad)
    ))

    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart training')
    print('**************')

    trainer.train(net, training_loader, val_loader, config)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)
