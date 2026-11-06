#
#      Class handling HFX3D dataset for KPConv-PyTorch.
#      Adapted from HuguesTHOMAS/KPConv-PyTorch (https://github.com/HuguesTHOMAS/KPConv-PyTorch), datasets/SensatUrban.py — reads H5 directly, no PLY conversion.
#
#      Dataset: urban building facades, 12 classes, xyz+rgb+intensity+labels
#      Split structure: train/ val/ test/ subdirs, one .h5 per building
#

import time
import numpy as np
import pickle
import torch
import math
import warnings
import h5py
from multiprocessing import Lock
from pathlib import Path

from datasets.common import PointCloudDataset, grid_subsampling
from torch.utils.data import Sampler, get_worker_info
from scipy.spatial import KDTree
from utils.config import bcolors


class HFX3DDataset(PointCloudDataset):
    """KPConv dataset class for HFX3D semantic segmentation."""

    def __init__(self, config, set='training', use_potentials=True, load_data=True):
        PointCloudDataset.__init__(self, 'HFX3D')

        self.label_to_names = {
            0: 'wall',
            1: 'window',
            2: 'door',
            3: 'balcony',
            4: 'vegetation',
            5: 'stairs',
            6: 'terrain',
            7: 'roof',
            8: 'blinds',
            9: 'other',
            10: 'column',
            11: 'arch',
        }

        self.init_labels()
        self.ignored_labels = np.array([])

        self.path = '/home/a/abdiaziz/data/HFX3D_Splits'
        self.dataset_task = 'cloud_segmentation'

        config.num_classes = self.num_classes - len(self.ignored_labels)
        config.dataset_task = self.dataset_task
        self.config = config
        self.set = set
        self.use_potentials = use_potentials

        # Trainer guard: validation_split must be in all_splits
        self.all_splits = ['training', 'validation']
        self.validation_split = 'validation'

        # Discover H5 files for this split
        if set == 'training':
            split_dir = Path(self.path) / 'train'
        elif set in ('validation', 'test', 'ERF'):
            split_dir = Path(self.path) / ('val' if set in ('validation', 'ERF') else 'test')
        else:
            raise ValueError('Unknown set for HFX3D: ' + set)

        self.files = sorted(split_dir.glob('*.h5'))
        self.cloud_names = [f.stem for f in self.files]
        # tester.py (shared across all KPConv datasets) does file_path.split('/'),
        # which fails on pathlib.Path objects -- keep self.files as plain strings.
        self.files = [str(f) for f in self.files]

        if len(self.files) == 0:
            raise FileNotFoundError('No .h5 files found in ' + str(split_dir))

        if set == 'training':
            self.epoch_n = config.epoch_steps * config.batch_num
        else:
            self.epoch_n = config.validation_size * config.batch_num

        if not load_data:
            return

        # Containers
        self.input_trees = []
        self.input_colors = []   # shape (N, 4): rgb + intensity
        self.input_labels = []
        self.pot_trees = []
        self.num_clouds = 0
        self.test_proj = []
        self.validation_labels = []

        self.load_subsampled_clouds()

        # Batch selection parameters
        self.batch_limit = torch.tensor([1], dtype=torch.float32)
        self.batch_limit.share_memory_()

        if use_potentials:
            self.potentials = []
            self.min_potentials = []
            self.argmin_potentials = []
            for tree in self.pot_trees:
                self.potentials.append(
                    torch.from_numpy(np.random.rand(tree.data.shape[0]) * 1e-3)
                )
                min_ind = int(torch.argmin(self.potentials[-1]))
                self.argmin_potentials.append(min_ind)
                self.min_potentials.append(float(self.potentials[-1][min_ind]))

            self.argmin_potentials = torch.from_numpy(
                np.array(self.argmin_potentials, dtype=np.int64)
            )
            self.min_potentials = torch.from_numpy(
                np.array(self.min_potentials, dtype=np.float64)
            )
            self.argmin_potentials.share_memory_()
            self.min_potentials.share_memory_()
            for pot in self.potentials:
                pot.share_memory_()

            self.worker_waiting = torch.tensor(
                [0 for _ in range(config.input_threads)], dtype=torch.int32
            )
            self.worker_waiting.share_memory_()
            self.epoch_inds = None
            self.epoch_i = 0
        else:
            self.potentials = None
            self.min_potentials = None
            self.argmin_potentials = None
            self.epoch_inds = torch.from_numpy(
                np.zeros((2, self.epoch_n), dtype=np.int64)
            )
            self.epoch_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
            self.epoch_i.share_memory_()
            self.epoch_inds.share_memory_()

        self.worker_lock = Lock()

        if self.set == 'ERF':
            self.batch_limit = torch.tensor([1], dtype=torch.float32)
            self.batch_limit.share_memory_()
            np.random.seed(42)

    def __len__(self):
        if self.use_potentials:
            return self.config.epoch_steps
        else:
            return int(np.ceil(self.config.epoch_steps / self.config.batch_num))

    def __getitem__(self, batch_i):
        if self.use_potentials:
            return self.potential_item(batch_i)
        else:
            return self.random_item(batch_i)

    def potential_item(self, batch_i, debug_workers=False):
        t = [time.time()]

        p_list, f_list, l_list, i_list, pi_list, ci_list, s_list, R_list = [], [], [], [], [], [], [], []
        batch_n = 0
        failed_attempts = 0

        info = get_worker_info()
        wid = info.id if info is not None else None

        while True:
            t += [time.time()]

            with self.worker_lock:
                cloud_ind = int(torch.argmin(self.min_potentials))
                point_ind = int(self.argmin_potentials[cloud_ind])

                pot_points = np.array(self.pot_trees[cloud_ind].data, copy=False)
                center_point = np.copy(pot_points[point_ind, :].reshape(1, -1))

                if self.set != 'ERF':
                    center_point += np.clip(
                        np.random.normal(scale=self.config.in_radius / 10, size=center_point.shape),
                        -self.config.in_radius / 2,
                        self.config.in_radius / 2,
                    )

                _pot_results = self.pot_trees[cloud_ind].query_ball_point(
                    center_point, r=self.config.in_radius
                )
                pot_inds = np.array(_pot_results[0], dtype=np.int64)
                _pot_pts = np.array(self.pot_trees[cloud_ind].data)[pot_inds]
                d2s = np.sum(np.square(_pot_pts - center_point), axis=1)

                if self.set != 'ERF':
                    tukeys = np.square(1 - d2s / np.square(self.config.in_radius))
                    tukeys[d2s > np.square(self.config.in_radius)] = 0
                    self.potentials[cloud_ind][pot_inds] += tukeys
                    min_ind = torch.argmin(self.potentials[cloud_ind])
                    self.min_potentials[[cloud_ind]] = self.potentials[cloud_ind][min_ind]
                    self.argmin_potentials[[cloud_ind]] = min_ind

            t += [time.time()]

            points = np.array(self.input_trees[cloud_ind].data, copy=False)
            input_inds = np.array(self.input_trees[cloud_ind].query_ball_point(
                center_point, r=self.config.in_radius
            )[0], dtype=np.int64)

            t += [time.time()]

            n = input_inds.shape[0]
            if n < 2:
                failed_attempts += 1
                if failed_attempts > 100 * self.config.batch_num:
                    raise ValueError('Dataset only contains empty input spheres')
                t += [time.time()]
                t += [time.time()]
                continue

            input_points = (points[input_inds] - center_point).astype(np.float32)
            # input_colors shape: (N, 4) — rgb + intensity
            input_features = self.input_colors[cloud_ind][input_inds].copy()

            if self.set in ('test', 'ERF'):
                input_labels = np.zeros(n, dtype=np.int32)
            else:
                input_labels = self.input_labels[cloud_ind][input_inds]
                input_labels = np.array([self.label_to_idx[l] for l in input_labels])

            t += [time.time()]

            input_points, scale, R = self.augmentation_transform(input_points)

            # Color augmentation: randomly zero out RGB only, not intensity
            if np.random.rand() > self.config.augment_color:
                input_features[:, :3] *= 0

            t += [time.time()]

            p_list.append(input_points)
            f_list.append(input_features)
            l_list.append(input_labels)
            pi_list.append(input_inds)
            i_list.append(point_ind)
            ci_list.append(cloud_ind)
            s_list.append(scale)
            R_list.append(R)

            batch_n += n
            if batch_n > int(self.batch_limit):
                break

        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        point_inds = np.array(i_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Assemble stacked features: [1, r, g, b, intensity] or subsets
        ones = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            stacked_features = ones
        elif self.config.in_features_dim == 3:
            stacked_features = stacked_points                        # xyz only
        elif self.config.in_features_dim == 4:
            # Both combos below are literally 4 channels, so in_features_dim
            # alone can't disambiguate (the model sizes its first layer from
            # this number, so it must stay 4 for either). config.feature_mode
            # picks between them; see train_HFX3D.py / test_HFX3D.py for how
            # it's set and persisted (NOT part of KPConv's own config
            # save/load, which only round-trips a fixed field list).
            if getattr(self.config, 'feature_mode', None) == 'xyz_intensity':
                stacked_features = np.hstack((stacked_points, features[:, 3:4]))  # xyz+intensity
            else:
                stacked_features = np.hstack((ones, features[:, :3]))   # 1+rgb (legacy default)
        elif self.config.in_features_dim == 5:
            stacked_features = np.hstack((ones, features))           # 1+rgb+intensity
        elif self.config.in_features_dim == 6:
            stacked_features = np.hstack((stacked_points, features[:, :3]))  # xyz+rgb
        elif self.config.in_features_dim == 7:
            stacked_features = np.hstack((stacked_points, features))  # xyz+rgb+intensity
        else:
            raise ValueError('Supported in_features_dim: 1, 3, 4, 5, 6, 7')

        t += [time.time()]

        input_list = self.segmentation_inputs(
            stacked_points, stacked_features, labels.astype(np.int32), stack_lengths
        )

        t += [time.time()]

        input_list += [scales, rots, cloud_inds, point_inds, input_inds]
        return input_list

    def random_item(self, batch_i):
        p_list, f_list, l_list, i_list, pi_list, ci_list, s_list, R_list = [], [], [], [], [], [], [], []
        batch_n = 0
        failed_attempts = 0

        while True:
            with self.worker_lock:
                cloud_ind = int(self.epoch_inds[0, self.epoch_i])
                point_ind = int(self.epoch_inds[1, self.epoch_i])
                self.epoch_i += 1
                if self.epoch_i >= int(self.epoch_inds.shape[1]):
                    self.epoch_i -= int(self.epoch_inds.shape[1])

            points = np.array(self.input_trees[cloud_ind].data, copy=False)
            center_point = np.copy(points[point_ind, :].reshape(1, -1))
            if self.set != 'ERF':
                center_point += np.clip(
                    np.random.normal(scale=self.config.in_radius / 10, size=center_point.shape),
                    -self.config.in_radius / 2,
                    self.config.in_radius / 2,
                )

            input_inds = np.array(self.input_trees[cloud_ind].query_ball_point(
                center_point, r=self.config.in_radius
            )[0], dtype=np.int64)
            n = input_inds.shape[0]

            if n < 2:
                failed_attempts += 1
                if failed_attempts > 100 * self.config.batch_num:
                    raise ValueError('Dataset only contains empty input spheres')
                continue

            input_points = (points[input_inds] - center_point).astype(np.float32)
            input_features = self.input_colors[cloud_ind][input_inds].copy()

            if self.set in ('test', 'ERF'):
                input_labels = np.zeros(n, dtype=np.int32)
            else:
                input_labels = self.input_labels[cloud_ind][input_inds]
                input_labels = np.array([self.label_to_idx[l] for l in input_labels])

            input_points, scale, R = self.augmentation_transform(input_points)
            if np.random.rand() > self.config.augment_color:
                input_features[:, :3] *= 0

            p_list.append(input_points)
            f_list.append(input_features)
            l_list.append(input_labels)
            pi_list.append(input_inds)
            i_list.append(point_ind)
            ci_list.append(cloud_ind)
            s_list.append(scale)
            R_list.append(R)

            batch_n += n
            if batch_n > int(self.batch_limit):
                break

        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        point_inds = np.array(i_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        ones = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            stacked_features = ones
        elif self.config.in_features_dim == 3:
            stacked_features = stacked_points                        # xyz only
        elif self.config.in_features_dim == 4:
            if getattr(self.config, 'feature_mode', None) == 'xyz_intensity':
                stacked_features = np.hstack((stacked_points, features[:, 3:4]))  # xyz+intensity
            else:
                stacked_features = np.hstack((ones, features[:, :3]))   # 1+rgb (legacy default)
        elif self.config.in_features_dim == 5:
            stacked_features = np.hstack((ones, features))
        elif self.config.in_features_dim == 6:
            stacked_features = np.hstack((stacked_points, features[:, :3]))  # xyz+rgb
        elif self.config.in_features_dim == 7:
            stacked_features = np.hstack((stacked_points, features))  # xyz+rgb+intensity
        else:
            raise ValueError('Supported in_features_dim: 1, 3, 4, 5, 6, 7')

        input_list = self.segmentation_inputs(
            stacked_points, stacked_features, labels.astype(np.int32), stack_lengths
        )
        input_list += [scales, rots, cloud_inds, point_inds, input_inds]
        return input_list

    def load_subsampled_clouds(self):
        dl = self.config.first_subsampling_dl
        cache_dir = Path(self.path) / 'input_{:.3f}'.format(dl)
        cache_dir.mkdir(exist_ok=True)

        for i, (file_path, cloud_name) in enumerate(zip(self.files, self.cloud_names)):
            t0 = time.time()
            tree_file = cache_dir / '{}_KDTree.pkl'.format(cloud_name)
            feat_file = cache_dir / '{}.npz'.format(cloud_name)

            if tree_file.exists() and feat_file.exists():
                print('\nFound cache for {:s} (dl={:.3f})'.format(cloud_name, dl))
                data = np.load(feat_file)
                sub_features = data['features']   # (N, 4): rgb+intensity
                sub_labels = data['labels']
                with open(tree_file, 'rb') as f:
                    search_tree = pickle.load(f)
            else:
                print('\nPreparing KDTree for {:s} (dl={:.3f})'.format(cloud_name, dl))
                with h5py.File(file_path, 'r') as hf:
                    xyz = hf['point_cloud/xyz'][:].astype(np.float32)
                    rgb = hf['point_cloud/rgb'][:].astype(np.float32)
                    intensity = hf['point_cloud/intensity'][:].astype(np.float32).reshape(-1, 1)
                    labels = hf['labels/semantic'][:].astype(np.int32).reshape(-1, 1)

                features = np.hstack([rgb, intensity])   # (N, 4)

                sub_xyz, sub_features, sub_labels = grid_subsampling(
                    xyz, features=features, labels=labels, sampleDl=dl
                )
                sub_labels = sub_labels.flatten().astype(np.int32)
                sub_features = sub_features.astype(np.float32)

                search_tree = KDTree(sub_xyz, leafsize=10)
                with open(tree_file, 'wb') as f:
                    pickle.dump(search_tree, f)
                np.savez(feat_file, features=sub_features, labels=sub_labels)

            self.input_trees.append(search_tree)
            self.input_colors.append(sub_features)
            self.input_labels.append(sub_labels)
            size_mb = sub_features.shape[0] * 4 * 7 * 1e-6
            print('{:.1f} MB loaded in {:.1f}s'.format(size_mb, time.time() - t0))

        # Coarse potential tree (downsampled version for choosing sphere centres)
        if self.use_potentials:
            print('\nPreparing potential trees')
            t0 = time.time()
            pot_dl = self.config.in_radius / 10

            for i, cloud_name in enumerate(self.cloud_names):
                coarse_tree_file = cache_dir / '{}_coarse_KDTree.pkl'.format(cloud_name)

                if coarse_tree_file.exists():
                    with open(coarse_tree_file, 'rb') as f:
                        coarse_tree = pickle.load(f)
                else:
                    sub_xyz = np.array(self.input_trees[i].data, copy=False)
                    coarse_pts = grid_subsampling(sub_xyz.astype(np.float32), sampleDl=pot_dl)
                    coarse_tree = KDTree(coarse_pts, leafsize=10)
                    with open(coarse_tree_file, 'wb') as f:
                        pickle.dump(coarse_tree, f)

                self.pot_trees.append(coarse_tree)

            print('Potential trees done in {:.1f}s'.format(time.time() - t0))

        # Reprojection indices for val/test evaluation
        self.num_clouds = len(self.input_trees)

        if self.set in ('validation', 'test'):
            print('\nPreparing reprojection indices')
            for i, (file_path, cloud_name) in enumerate(zip(self.files, self.cloud_names)):
                t0 = time.time()
                proj_file = cache_dir / '{}_proj.pkl'.format(cloud_name)

                if proj_file.exists():
                    with open(proj_file, 'rb') as f:
                        proj_inds, val_labels = pickle.load(f)
                else:
                    with h5py.File(file_path, 'r') as hf:
                        full_xyz = hf['point_cloud/xyz'][:].astype(np.float32)
                        val_labels = hf['labels/semantic'][:].astype(np.int32)

                    _, idxs = self.input_trees[i].query(full_xyz, k=1)
                    proj_inds = np.squeeze(idxs).astype(np.int32)
                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, val_labels], f)

                self.test_proj.append(proj_inds)
                self.validation_labels.append(val_labels)
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

        print()

    def load_evaluation_points(self, file_path):
        with h5py.File(file_path, 'r') as hf:
            xyz = hf['point_cloud/xyz'][:].astype(np.float32)
        return xyz


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class HFX3DSampler(Sampler):

    def __init__(self, dataset: HFX3DDataset):
        Sampler.__init__(self, dataset)
        self.dataset = dataset
        self.N = (
            dataset.config.epoch_steps
            if dataset.set == 'training'
            else dataset.config.validation_size
        )

    def __iter__(self):
        if not self.dataset.use_potentials:
            self.dataset.epoch_i *= 0
            self.dataset.epoch_inds *= 0

            num_centers = self.N * self.dataset.config.batch_num
            random_pick_n = int(np.ceil(num_centers / self.dataset.config.num_classes))

            all_epoch_inds = np.zeros((2, 0), dtype=np.int64)
            for label_ind, label in enumerate(self.dataset.label_values):
                if label not in self.dataset.ignored_labels:
                    all_label_indices = []
                    for cloud_ind, cloud_labels in enumerate(self.dataset.input_labels):
                        label_indices = np.where(np.equal(cloud_labels, label))[0]
                        all_label_indices.append(
                            np.vstack((
                                np.full(label_indices.shape, cloud_ind, dtype=np.int64),
                                label_indices,
                            ))
                        )
                    all_label_indices = np.hstack(all_label_indices)
                    N_inds = all_label_indices.shape[1]

                    if N_inds < random_pick_n:
                        chosen = np.zeros((2, 0), dtype=np.int64)
                        while chosen.shape[1] < random_pick_n:
                            chosen = np.hstack((chosen, all_label_indices[:, np.random.permutation(N_inds)]))
                        warnings.warn('Class {:d} undersampled, repeating indices'.format(label))
                    elif N_inds < 50 * random_pick_n:
                        rand_inds = np.random.choice(N_inds, size=random_pick_n, replace=False)
                        chosen = all_label_indices[:, rand_inds]
                    else:
                        chosen = np.zeros((2, 0), dtype=np.int64)
                        while chosen.shape[1] < random_pick_n:
                            rand_inds = np.unique(np.random.choice(N_inds, size=2 * random_pick_n, replace=True))
                            chosen = np.hstack((chosen, all_label_indices[:, rand_inds]))
                        chosen = chosen[:, :random_pick_n]

                    all_epoch_inds = np.hstack((all_epoch_inds, chosen))

            random_order = np.random.permutation(all_epoch_inds.shape[1])[:num_centers]
            all_epoch_inds = all_epoch_inds[:, random_order].astype(np.int64)
            self.dataset.epoch_inds += torch.from_numpy(all_epoch_inds)

        for i in range(self.N):
            yield i

    def __len__(self):
        return self.N

    def calibration(self, dataloader, untouched_ratio=0.9, verbose=False, force_redo=False):
        print('\nStarting Calibration (use verbose=True for more details)')
        t0 = time.time()
        redo = force_redo

        # Batch limit
        batch_lim_file = str(Path(self.dataset.path) / 'batch_limits.pkl')
        if Path(batch_lim_file).exists():
            with open(batch_lim_file, 'rb') as f:
                batch_lim_dict = pickle.load(f)
        else:
            batch_lim_dict = {}

        sampler_method = 'potentials' if self.dataset.use_potentials else 'random'
        key = '{:s}_{:.3f}_{:.3f}_{:d}'.format(
            sampler_method,
            self.dataset.config.in_radius,
            self.dataset.config.first_subsampling_dl,
            self.dataset.config.batch_num,
        )
        if not redo and key in batch_lim_dict:
            self.dataset.batch_limit[0] = batch_lim_dict[key]
        else:
            redo = True

        if verbose:
            print('\nCheck batch limit dictionary')
            color = bcolors.OKGREEN if key in batch_lim_dict else bcolors.FAIL
            v = str(int(batch_lim_dict[key])) if key in batch_lim_dict else '?'
            print('{:}"{:s}": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        # Neighbor limits
        neighb_lim_file = str(Path(self.dataset.path) / 'neighbors_limits.pkl')
        if Path(neighb_lim_file).exists():
            with open(neighb_lim_file, 'rb') as f:
                neighb_lim_dict = pickle.load(f)
        else:
            neighb_lim_dict = {}

        neighb_limits = []
        for layer_ind in range(self.dataset.config.num_layers):
            dl = self.dataset.config.first_subsampling_dl * (2 ** layer_ind)
            r = dl * (
                self.dataset.config.deform_radius
                if self.dataset.config.deform_layers[layer_ind]
                else self.dataset.config.conv_radius
            )
            key_n = '{:.3f}_{:.3f}'.format(dl, r)
            if key_n in neighb_lim_dict:
                neighb_limits.append(neighb_lim_dict[key_n])

        if not redo and len(neighb_limits) == self.dataset.config.num_layers:
            self.dataset.neighborhood_limits = neighb_limits
        else:
            redo = True

        if redo:
            hist_n = int(np.ceil(4 / 3 * np.pi * (self.dataset.config.deform_radius + 1) ** 3))
            neighb_hists = np.zeros((self.dataset.config.num_layers, hist_n), dtype=np.int32)

            estim_b = 0
            target_b = self.dataset.config.batch_num
            expected_N = 10000
            low_pass_T = 100
            Kp = expected_N / 200
            Ki = 0.001 * Kp
            Kd = 5 * Kp
            finer = False
            stabilized = False
            smooth_errors = []
            converge_threshold = 0.1
            last_display = time.time()
            i = 0
            breaking = False
            error_I = 0
            error_D = 0
            last_error = 0

            sample_batches = 999
            for epoch in range((sample_batches // self.N) + 1):
                for batch_i, batch in enumerate(dataloader):
                    counts = [
                        np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1)
                        for neighb_mat in batch.neighbors
                    ]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)

                    b = len(batch.cloud_inds)
                    estim_b += (b - estim_b) / low_pass_T
                    error = target_b - b
                    error_I += error
                    error_D = error - last_error
                    last_error = error

                    smooth_errors.append(target_b - estim_b)
                    if len(smooth_errors) > 30:
                        smooth_errors = smooth_errors[1:]

                    self.dataset.batch_limit += Kp * error + Ki * error_I + Kd * error_D

                    if not stabilized and self.dataset.batch_limit < 0:
                        Kp *= 0.1
                        Ki *= 0.1
                        Kd *= 0.1
                        stabilized = True

                    if not finer and np.abs(estim_b - target_b) < 1:
                        low_pass_T = 100
                        finer = True

                    if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                        breaking = True
                        break

                    i += 1
                    t = time.time()
                    if verbose and (t - last_display) > 1.0:
                        last_display = t
                        print('Step {:5d}  estim_b={:5.2f}  batch_limit={:7d}'.format(
                            i, estim_b, int(self.dataset.batch_limit)))

                if breaking:
                    break

            if not breaking:
                print('WARNING: Calibration did not converge. Check in_radius and first_subsampling_dl.')

            cumsum = np.cumsum(neighb_hists.T, axis=0)
            percentiles = np.sum(cumsum < (untouched_ratio * cumsum[hist_n - 1, :]), axis=0)
            self.dataset.neighborhood_limits = percentiles

            batch_lim_dict[key] = float(self.dataset.batch_limit)
            with open(batch_lim_file, 'wb') as f:
                pickle.dump(batch_lim_dict, f)

            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2 ** layer_ind)
                r = dl * (
                    self.dataset.config.deform_radius
                    if self.dataset.config.deform_layers[layer_ind]
                    else self.dataset.config.conv_radius
                )
                key_n = '{:.3f}_{:.3f}'.format(dl, r)
                neighb_lim_dict[key_n] = self.dataset.neighborhood_limits[layer_ind]
            with open(neighb_lim_file, 'wb') as f:
                pickle.dump(neighb_lim_dict, f)

        print('Calibration done in {:.1f}s\n'.format(time.time() - t0))


# ---------------------------------------------------------------------------
# Custom batch
# ---------------------------------------------------------------------------

class HFX3DCustomBatch:
    """Batch container with memory pinning for HFX3D."""

    def __init__(self, input_list):
        input_list = input_list[0]
        L = (len(input_list) - 7) // 5

        ind = 0
        self.points = [torch.from_numpy(a) for a in input_list[ind:ind + L]]; ind += L
        self.neighbors = [torch.from_numpy(a) for a in input_list[ind:ind + L]]; ind += L
        self.pools = [torch.from_numpy(a) for a in input_list[ind:ind + L]]; ind += L
        self.upsamples = [torch.from_numpy(a) for a in input_list[ind:ind + L]]; ind += L
        self.lengths = [torch.from_numpy(a) for a in input_list[ind:ind + L]]; ind += L
        self.features = torch.from_numpy(input_list[ind]); ind += 1
        self.labels = torch.from_numpy(input_list[ind]); ind += 1
        self.scales = torch.from_numpy(input_list[ind]); ind += 1
        self.rots = torch.from_numpy(input_list[ind]); ind += 1
        self.cloud_inds = torch.from_numpy(input_list[ind]); ind += 1
        self.center_inds = torch.from_numpy(input_list[ind]); ind += 1
        self.input_inds = torch.from_numpy(input_list[ind])

    def pin_memory(self):
        self.points = [t.pin_memory() for t in self.points]
        self.neighbors = [t.pin_memory() for t in self.neighbors]
        self.pools = [t.pin_memory() for t in self.pools]
        self.upsamples = [t.pin_memory() for t in self.upsamples]
        self.lengths = [t.pin_memory() for t in self.lengths]
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.cloud_inds = self.cloud_inds.pin_memory()
        self.center_inds = self.center_inds.pin_memory()
        self.input_inds = self.input_inds.pin_memory()
        return self

    def to(self, device):
        self.points = [t.to(device) for t in self.points]
        self.neighbors = [t.to(device) for t in self.neighbors]
        self.pools = [t.to(device) for t in self.pools]
        self.upsamples = [t.to(device) for t in self.upsamples]
        self.lengths = [t.to(device) for t in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.cloud_inds = self.cloud_inds.to(device)
        self.center_inds = self.center_inds.to(device)
        self.input_inds = self.input_inds.to(device)
        return self

    def unstack_points(self, layer=None):
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError('Unknown element: ' + element_name)

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):
            if layer is None or layer == layer_i:
                i0 = 0
                p_list = []
                lengths = self.lengths[layer_i + 1] if element_name == 'pools' else self.lengths[layer_i]
                for b_i, length in enumerate(lengths):
                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])
                    i0 += length
                    p_list.append(elem.numpy() if to_numpy else elem)
                if layer == layer_i:
                    return p_list
                all_p_list.append(p_list)
        return all_p_list


def HFX3DCollate(batch_data):
    return HFX3DCustomBatch(batch_data)
