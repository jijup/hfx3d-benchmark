"""
PointNet++ semantic segmentation training on HFX3D dataset.

Usage (inside SIF):
  python train_semseg_hfx3d.py \
      --model pointnet2_sem_seg \
      --data_root /home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits \
      --epoch 64 \
      --batch_size 16 \
      --npoint 4096 \
      --log_dir hfx3d_run1

Based on the original train_semseg.py by Benny (Nov 2019), adapted for HFX3D.
"""

import argparse
import os
import datetime
import logging
import importlib
import shutil
import sys
import time

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from data_utils.HFX3DDataLoader import HFX3DDataset, NUM_CLASSES, CLASS_NAMES
import provider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))

seg_label_to_cat = {i: name for i, name in enumerate(CLASS_NAMES)}


def inplace_relu(m):
    if m.__class__.__name__.find("ReLU") != -1:
        m.inplace = True


def parse_args():
    parser = argparse.ArgumentParser("PointNet++ HFX3D SemSeg")
    parser.add_argument("--model", type=str, default="pointnet2_sem_seg",
                        help="model name [default: pointnet2_sem_seg]")
    parser.add_argument("--data_root", type=str,
                        default="/home/abdiaziz/projects/def-jijup/abdiaziz/HFX3D_Splits",
                        help="Path to HFX3D_Splits directory")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size during training [default: 16]")
    parser.add_argument("--epoch", type=int, default=200,
                        help="Number of epochs [default: 64]")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Initial learning rate [default: 0.001]")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="Adam or SGD [default: Adam]")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Log subdirectory name [default: timestamp]")
    parser.add_argument("--decay_rate", type=float, default=1e-4,
                        help="Weight decay [default: 1e-4]")
    parser.add_argument("--npoint", type=int, default=4096,
                        help="Points per sample [default: 4096]")
    parser.add_argument("--step_size", type=int, default=10,
                        help="LR decay step (epochs) [default: 10]")
    parser.add_argument("--lr_decay", type=float, default=0.7,
                        help="LR decay factor [default: 0.7]")
    parser.add_argument("--sample_rate", type=float, default=1.0,
                        help="Dataset sample rate [default: 1.0]")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main(args):
    # ------------------------------------------------------------------ dirs
    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    experiment_dir = Path("./log/sem_seg_hfx3d")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.log_dir if args.log_dir else timestr
    experiment_dir = experiment_dir / run_name
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------------- logger
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(str(log_dir / f"{args.model}.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    def log_string(s):
        logger.info(s)
        print(s)

    log_string("PARAMETERS: " + str(args))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --------------------------------------------------------------- datasets
    print("Loading training data ...")
    TRAIN_DATASET = HFX3DDataset(
        split="train",
        data_root=args.data_root,
        num_point=args.npoint,
        sample_rate=args.sample_rate,
    )
    print("Loading test data ...")
    TEST_DATASET = HFX3DDataset(
        split="val",
        data_root=args.data_root,
        num_point=args.npoint,
        sample_rate=args.sample_rate,
    )

    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True,
        worker_init_fn=lambda x: np.random.seed(args.seed + x),
    )
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=True,
    )
    # weights removed — unweighted CE for benchmark consistency

    log_string(f"Training samples : {len(TRAIN_DATASET)}")
    log_string(f"Test samples     : {len(TEST_DATASET)}")

    # ------------------------------------------------------------------ model
    MODEL = importlib.import_module(args.model)
    shutil.copy(f"models/{args.model}.py", str(experiment_dir))
    shutil.copy("models/pointnet2_utils.py", str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    # Use DataParallel if multiple GPUs available
    num_gpus = torch.cuda.device_count()
    log_string(f"Using {num_gpus} GPU(s)")
    if num_gpus > 1:
        classifier = torch.nn.DataParallel(classifier)

    def weights_init(m):
        cn = m.__class__.__name__
        if cn.find("Conv2d") != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif cn.find("Linear") != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(checkpoints_dir / "best_model.pth"))
        start_epoch = checkpoint["epoch"]
        state_dict = checkpoint["model_state_dict"]
        # Handle DataParallel wrapping
        if num_gpus > 1:
            classifier.module.load_state_dict(state_dict)
        else:
            classifier.load_state_dict(state_dict)
        log_string("Loaded pretrained model")
    except Exception:
        log_string("No existing model — training from scratch")
        start_epoch = 0
        classifier.apply(weights_init)

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(), lr=args.learning_rate, momentum=0.9
        )

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECAY = 0.5
    best_iou = 0.0

    # ---------------------------------------------------------------- training
    for epoch in range(start_epoch, args.epoch):
        log_string(f"\n**** Epoch {epoch + 1}/{args.epoch} ****")

        lr = max(
            args.learning_rate * (args.lr_decay ** (epoch // args.step_size)),
            LEARNING_RATE_CLIP,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        log_string(f"Learning rate: {lr:.6f}")

        momentum = max(
            MOMENTUM_ORIGINAL * (MOMENTUM_DECAY ** (epoch // args.step_size)), 0.01
        )
        classifier.apply(lambda m: bn_momentum_adjust(m, momentum))

        # ---- train loop
        classifier.train()
        total_correct = 0
        total_seen = 0
        loss_sum = 0.0

        for points, target in tqdm(trainDataLoader, desc="Train", smoothing=0.9):
            optimizer.zero_grad()

            # Z-axis random rotation augmentation
            pts_np = points.numpy()
            pts_np[:, :, :3] = provider.rotate_point_cloud_z(pts_np[:, :, :3])
            points = torch.from_numpy(pts_np)

            points = points.float().cuda()          # (B, N, 9)
            target = target.long().cuda()           # (B, N)
            points = points.transpose(2, 1)         # (B, 9, N)

            seg_pred, trans_feat = classifier(points)  # (B, N, C)
            seg_pred_flat = seg_pred.contiguous().view(-1, NUM_CLASSES)
            target_flat = target.view(-1)

            loss = criterion(seg_pred_flat, target_flat, trans_feat, None)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred_flat.cpu().data.max(1)[1].numpy()
            batch_label = target_flat.cpu().numpy()
            total_correct += np.sum(pred_choice == batch_label)
            total_seen += pred_choice.shape[0]
            loss_sum += loss.item()

        n_batches = len(trainDataLoader)
        log_string(f"Train mean loss  : {loss_sum / n_batches:.4f}")
        log_string(f"Train accuracy   : {total_correct / total_seen:.4f}")

        if (epoch + 1) % 5 == 0:
            savepath = str(checkpoints_dir / "model.pth")
            state = {
                "epoch": epoch,
                "model_state_dict": (
                    classifier.module.state_dict()
                    if num_gpus > 1 else classifier.state_dict()
                ),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            import shutil as _shl  # _PN2_PERIODIC_SAVE_
            _tmp = '/tmp/pn2_ckpt_{}.pth'.format(os.getpid())
            torch.save(state, _tmp)
            for _att in range(10):
                try: _shl.copy2(_tmp, savepath); break
                except Exception as _e:
                    if _att < 9:
                        log_string(f"Copy attempt {_att+1}/10 failed: {_e}. Retrying...")
                        time.sleep(10)
                    else: os.remove(_tmp); raise
            os.remove(_tmp)
            log_string(f"Checkpoint saved to {savepath}")

        # ---- eval loop
        classifier.eval()
        with torch.no_grad():
            total_correct = 0
            total_seen = 0
            loss_sum = 0.0
            total_seen_class = np.zeros(NUM_CLASSES)
            total_correct_class = np.zeros(NUM_CLASSES)
            total_iou_denom_class = np.zeros(NUM_CLASSES)

            log_string(f"---- Epoch {epoch + 1:03d} EVALUATION ----")
            for points, target in tqdm(testDataLoader, desc="Eval", smoothing=0.9):
                points = points.float().cuda().transpose(2, 1)
                target = target.long().cuda()

                seg_pred, trans_feat = classifier(points)
                seg_pred_flat = seg_pred.contiguous().view(-1, NUM_CLASSES)
                target_flat = target.view(-1)

                loss = criterion(seg_pred_flat, target_flat, trans_feat, None)
                loss_sum += loss.item()

                pred_val = seg_pred.cpu().numpy().argmax(axis=2)   # (B, N)
                batch_label = target.cpu().numpy()                  # (B, N)

                total_correct += np.sum(pred_val == batch_label)
                total_seen += batch_label.size

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum(batch_label == l)
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_denom_class[l] += np.sum((pred_val == l) | (batch_label == l))

            n_batches_eval = len(testDataLoader)
            iou_per_class = total_correct_class / (total_iou_denom_class + 1e-6)
            mIoU = np.mean(iou_per_class)
            oa = total_correct / (total_seen + 1e-9)
            # precision = TP / (TP+FP); TP+FP = iou_denom - (seen - correct) = predicted count
            predicted_class = total_iou_denom_class - (total_seen_class - total_correct_class)
            precision_per_class = total_correct_class / (predicted_class + 1e-6)
            recall_per_class = total_correct_class / (total_seen_class + 1e-6)
            f1_per_class = (2 * precision_per_class * recall_per_class
                            / (precision_per_class + recall_per_class + 1e-9))
            mPrecision = np.mean(precision_per_class)
            mRecall = np.mean(recall_per_class)
            mF1 = np.mean(f1_per_class)

            log_string(f"Eval mean loss   : {loss_sum / n_batches_eval:.4f}")
            log_string(f"Eval OA/mP/mR/mF1/mIoU: {oa:.4f} / {mPrecision:.4f} / {mRecall:.4f} / {mF1:.4f} / {mIoU:.4f}")

            iou_str = "------- Per-class IoU/Precision/Recall/F1 -------\n"
            for l in range(NUM_CLASSES):
                iou_str += (f"  {CLASS_NAMES[l]:<12s}  IoU:{iou_per_class[l]:.4f}  "
                            f"P:{precision_per_class[l]:.4f}  R:{recall_per_class[l]:.4f}  "
                            f"F1:{f1_per_class[l]:.4f}\n")
            log_string(iou_str)

            if mIoU >= best_iou:
                best_iou = mIoU
                savepath = str(checkpoints_dir / "best_model.pth")
                state = {
                    "epoch": epoch,
                    "class_avg_iou": mIoU,
                    "model_state_dict": (
                        classifier.module.state_dict()
                        if num_gpus > 1 else classifier.state_dict()
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                import shutil as _shl  # _PN2_BEST_SAVE_
                _tmp = '/tmp/pn2_best_{}.pth'.format(os.getpid())
                torch.save(state, _tmp)
                for _att in range(10):
                    try: _shl.copy2(_tmp, savepath); break
                    except Exception as _e:
                        if _att < 9:
                            log_string(f"Copy attempt {_att+1}/10 failed: {_e}. Retrying...")
                            time.sleep(10)
                        else: os.remove(_tmp); raise
                os.remove(_tmp)
                log_string(f"Best model saved (mIoU={best_iou:.4f})")

        log_string(f"Best mIoU so far : {best_iou:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
