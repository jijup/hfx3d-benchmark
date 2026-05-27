"""
RandLA-Net model adapted for HFX3D.

Changes vs original RandLANet.py
---------------------------------
* fc0 accepts cfg.input_feature_dim (7) channels instead of hard-coded 3
* class_weights sourced from cfg.class_weights (set by HFX3DDataset)
* sklearn replaced with numpy for confusion matrix (avoids missing dependency)
* No ignored labels (all 12 HFX3D classes are valid)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_utils as pt_utils
from helper_hfx3d import ConfigHFX3D, DataProcessing as DP


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        in_feat = getattr(config, 'input_feature_dim', 3)
        self.fc0 = pt_utils.Conv1d(in_feat, 8, kernel_size=1, bn=True)

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(config.num_layers):
            d_out = config.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out

        d_out = d_in
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

        # Precompute encoder output channels so decoder dims are correct.
        # f_encoder_list has num_layers+1 entries:
        #   [0]       = full-res after layer 0  (2*d_out[0])
        #   [1..L]    = sub-sampled after each layer  (2*d_out[i])
        enc_out_ch = [2 * config.d_out[0]]  # index 0
        for i in range(config.num_layers):
            enc_out_ch.append(2 * config.d_out[i])  # indices 1..num_layers

        self.decoder_blocks = nn.ModuleList()
        for j in range(config.num_layers):
            # Skip-connection from encoder list at position -(j+2)
            skip_ch = enc_out_ch[-(j + 2)]
            d_in  = d_out + skip_ch
            d_out = skip_ch
            self.decoder_blocks.append(
                pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True))

        self.fc1 = pt_utils.Conv2d(d_out, 64, kernel_size=(1, 1), bn=True)
        self.fc2 = pt_utils.Conv2d(64, 32,    kernel_size=(1, 1), bn=True)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = pt_utils.Conv2d(32, config.num_classes,
                                   kernel_size=(1, 1), bn=False, activation=None)

    def forward(self, end_points):
        features = end_points['features']          # (B, C, N)
        features = self.fc0(features)
        features = features.unsqueeze(dim=3)       # (B, C, N, 1)

        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_enc = self.dilated_res_blocks[i](
                features, end_points['xyz'][i], end_points['neigh_idx'][i])
            f_sub = self.random_sample(f_enc, end_points['sub_idx'][i])
            features = f_sub
            if i == 0:
                f_encoder_list.append(f_enc)
            f_encoder_list.append(f_sub)

        features = self.decoder_0(f_encoder_list[-1])

        for j in range(self.config.num_layers):
            f_interp = self.nearest_interpolation(
                features, end_points['interp_idx'][-j - 1])
            f_dec = self.decoder_blocks[j](
                torch.cat([f_encoder_list[-j - 2], f_interp], dim=1))
            features = f_dec

        features = self.fc1(features)
        features = self.fc2(features)
        features = self.dropout(features)
        features = self.fc3(features)
        f_out = features.squeeze(3)       # (B, num_classes, N)

        end_points['logits'] = f_out
        return end_points

    @staticmethod
    def random_sample(feature, pool_idx):
        feature = feature.squeeze(dim=3)
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        B = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(B, -1)
        pool_features = torch.gather(
            feature, 2, pool_idx.unsqueeze(1).repeat(1, d, 1))
        pool_features = pool_features.reshape(B, d, -1, num_neigh)
        return pool_features.max(dim=3, keepdim=True)[0]

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        feature = feature.squeeze(dim=3)
        B = interp_idx.shape[0]
        up_N = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(B, up_N)
        interp = torch.gather(
            feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        return interp.unsqueeze(3)


# ---------------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------------

def compute_loss(end_points, config):
    logits = end_points['logits']            # (B, C, N)
    labels = end_points['labels']            # (B, N)

    logits = logits.transpose(1, 2).reshape(-1, config.num_classes)
    labels = labels.reshape(-1)

    # No ignored labels for HFX3D — all 12 classes are valid
    valid_logits = logits
    valid_labels = labels

    loss = get_loss(valid_logits, valid_labels, config.class_weights)
    end_points['valid_logits'] = valid_logits
    end_points['valid_labels'] = valid_labels
    end_points['loss'] = loss
    return loss, end_points


def get_loss(logits, labels, pre_cal_weights):
    if pre_cal_weights is not None:
        # pre_cal_weights may be (1, C) — squeeze to (C,)
        w = np.squeeze(pre_cal_weights)
        class_weights = torch.from_numpy(w).float().to(logits.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss(reduction='mean')
    return criterion(logits, labels)


def compute_acc(end_points):
    logits = end_points['valid_logits']
    labels = end_points['valid_labels']
    pred   = logits.max(dim=1)[1]
    acc    = (pred == labels).float().mean()
    end_points['acc'] = acc
    return acc, end_points


# ---------------------------------------------------------------------------
# IoU calculator (pure numpy — no sklearn dependency)
# ---------------------------------------------------------------------------

class IoUCalculator:
    def __init__(self, cfg):
        self.num_classes = cfg.num_classes
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes),
                                    dtype=np.int64)

    def add_data(self, end_points):
        logits = end_points['valid_logits']
        labels = end_points['valid_labels']
        pred   = logits.max(dim=1)[1].detach().cpu().numpy()
        gt     = labels.detach().cpu().numpy()

        # Vectorised confusion-matrix accumulation
        mask = (gt >= 0) & (gt < self.num_classes)
        combined = self.num_classes * gt[mask].astype(np.int64) + pred[mask].astype(np.int64)
        bincount = np.bincount(combined,
                               minlength=self.num_classes ** 2)
        self.conf_matrix += bincount.reshape(self.num_classes, self.num_classes)

    def compute_iou(self):
        tp  = np.diag(self.conf_matrix).astype(np.float64)
        fp  = self.conf_matrix.sum(axis=0).astype(np.float64) - tp
        fn  = self.conf_matrix.sum(axis=1).astype(np.float64) - tp
        iou = tp / (tp + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall    = tp / (tp + fn + 1e-6)
        f1        = 2 * precision * recall / (precision + recall + 1e-6)
        oa        = float(tp.sum() / (self.conf_matrix.sum() + 1e-6))
        mean_iou       = float(np.mean(iou))
        mean_precision = float(np.mean(precision))
        mean_recall    = float(np.mean(recall))
        mean_f1        = float(np.mean(f1))
        return mean_iou, mean_precision, mean_recall, mean_f1, oa, iou.tolist(), precision.tolist(), recall.tolist(), f1.tolist()


# ---------------------------------------------------------------------------
# Sub-modules (unchanged from original)
# ---------------------------------------------------------------------------

class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.mlp1     = pt_utils.Conv2d(d_in,    d_out // 2,   kernel_size=(1, 1), bn=True)
        self.lfa      = Building_block(d_out)
        self.mlp2     = pt_utils.Conv2d(d_out,   d_out * 2,    kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in,    d_out * 2,    kernel_size=(1, 1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f = self.mlp1(feature)
        f = self.lfa(xyz, f, neigh_idx)
        f = self.mlp2(f)
        s = self.shortcut(feature)
        return F.leaky_relu(f + s, negative_slope=0.2)


class Building_block(nn.Module):
    def __init__(self, d_out):
        super().__init__()
        self.mlp1        = pt_utils.Conv2d(10, d_out // 2, kernel_size=(1, 1), bn=True)
        self.att_pool_1  = Att_pooling(d_out,       d_out // 2)
        self.mlp2        = pt_utils.Conv2d(d_out // 2, d_out // 2, kernel_size=(1, 1), bn=True)
        self.att_pool_2  = Att_pooling(d_out,       d_out)

    def forward(self, xyz, feature, neigh_idx):
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)   # (B, N, k, 10)
        f_xyz = f_xyz.permute(0, 3, 1, 2)                    # (B, 10, N, k)
        f_xyz = self.mlp1(f_xyz)

        f_nb  = self.gather_neighbour(
            feature.squeeze(-1).permute(0, 2, 1), neigh_idx)  # (B, N, k, C)
        f_nb  = f_nb.permute(0, 3, 1, 2)                      # (B, C, N, k)
        f_cat = torch.cat([f_nb, f_xyz], dim=1)
        f_agg = self.att_pool_1(f_cat)

        f_xyz = self.mlp2(f_xyz)
        f_nb  = self.gather_neighbour(
            f_agg.squeeze(-1).permute(0, 2, 1), neigh_idx)
        f_nb  = f_nb.permute(0, 3, 1, 2)
        f_cat = torch.cat([f_nb, f_xyz], dim=1)
        f_agg = self.att_pool_2(f_cat)
        return f_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        nb_xyz   = self.gather_neighbour(xyz, neigh_idx)            # (B, N, k, 3)
        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)
        rel_xyz  = xyz_tile - nb_xyz
        rel_dist = torch.sqrt(torch.sum(rel_xyz ** 2, dim=-1, keepdim=True) + 1e-10)
        return torch.cat([rel_dist, rel_xyz, xyz_tile, nb_xyz], dim=-1)  # (B,N,k,10)

    @staticmethod
    def gather_neighbour(pc, neigh_idx):
        B, N, d = pc.shape
        idx_flat = neigh_idx.reshape(B, -1)
        feat = torch.gather(pc, 1,
                            idx_flat.unsqueeze(-1).expand(-1, -1, d))
        return feat.reshape(B, N, neigh_idx.shape[-1], d)


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc  = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

    def forward(self, feature_set):
        scores = F.softmax(self.fc(feature_set), dim=3)
        f_agg  = (feature_set * scores).sum(dim=3, keepdim=True)
        return self.mlp(f_agg)
