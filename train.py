import argparse
import torch
import os

from torch import optim
from torch.utils.data import DataLoader

from src.datagen import TrainingDataset, TestingDataset, custom_collate_fn
from src.models.network1 import *
from src.models.build import *
from src.utils.loss import FocalDiceloss
from src.utils.utils import get_logger, setup_seeds
from src.utils.metrics import SegMetrics
from src.config import Config

import time
from tqdm import tqdm
import numpy as np
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Train SAM-based segmentation model")

    parser.add_argument("--run_name", type=str, default="train",
                        help="Unique name for this training run (used for saving logs/models)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Total number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training and evaluation")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'],
                        help="List of evaluation metrics to compute")
    parser.add_argument("--device", type=str, default='cuda:0',
                        help="Device identifier for training, e.g., 'cuda:0' or 'cpu'")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of subprocesses for data loading")

    return parser.parse_args()
            

def move_batch_to_device(batch, device):
    moved_batch = []
    for sample in batch:
        sample_on_device = {}
        for key, val in sample.items():
            if key in ['image', 'labels', 'boxes']:
                sample_on_device[key] = val.float().to(device)
            else:
                sample_on_device[key] = val
        moved_batch.append(sample_on_device)
    return moved_batch


def run_one_training_epoch(args, model, optimizer, data_loader, epoch_num, loss_fn, logger):
    model.train()
    data_iter = tqdm(data_loader)
    total_loss_values = []
    cumulative_metrics = [0] * len(args.metrics)

    for batch_idx, batch in enumerate(data_iter):
        current_batch_size = len(batch)
        batch = move_batch_to_device(batch, args.device)

        model_outputs = model.train_forward(batch)
        pred_masks = torch.cat([output['masks'] for output in model_outputs], dim=0)
        true_masks = torch.cat([item['labels'] for item in batch], dim=0)

        focal, dice = loss_fn(pred_masks, true_masks)
        combined_loss = focal + 20 * dice
        combined_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        batch_metrics = sum([SegMetrics(model_outputs[i]['masks'], batch[i]['labels'], args.metrics)
                             for i in range(current_batch_size)]) / current_batch_size

        if batch_idx % 50 == 0 or batch_idx == 0:
            print(f"[Epoch {epoch_num+1}] Batch {batch_idx+1} - Metrics: {batch_metrics}")
            logger.info(f"[Epoch {epoch_num+1}] Batch {batch_idx+1}, Loss: {combined_loss.item()}, "
                        f"Focal: {focal.item()}, Dice: {dice.item()}, Metrics: {batch_metrics}")

        cumulative_metrics = [cumulative_metrics[i] + batch_metrics[i] for i in range(len(args.metrics))]
        total_loss_values.append(combined_loss.item())

        data_iter.set_postfix(train_loss=combined_loss.item(), gpu=args.device)

    return total_loss_values, cumulative_metrics

def evaluate_one_epoch(cfg, model, loader, epoch_id):
    model.eval()
    loader = tqdm(loader)
    accumulated_metrics = [0] * len(cfg.metrics)

    for step, batch_data in enumerate(loader):
        batch_size = len(batch_data)
        batch_data = move_batch_to_device(batch_data, cfg.device)
        predictions = model.infer(batch_data)

        batch_metrics = sum(
            [SegMetrics(predictions[i]["masks"], batch_data[i]["labels"], cfg.metrics) 
             for i in range(batch_size)]
        ) / batch_size

        if (step + 1) % 50 == 0:
            print(f"[Eval Epoch {epoch_id+1}] Step {step+1} - Metrics: {batch_metrics}")

        for i in range(len(cfg.metrics)):
            accumulated_metrics[i] += batch_metrics[i]

    return accumulated_metrics

def main(cfg):
    setup_seeds()

    model = MultiStreamSegmentor().to(cfg.device)
    path = r"C:\Users\UAB\CK_WorkPlace\PhD\Prostate\samunet\weights\proposed\uab\resnet\model_train\train\epoch15_model.pth"
    model = model_registry["resSAM"](need_ori_checkpoint=False, model_checkpoint=path).to('cuda')
    print('loaded pretrained model!')
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.5)
    criterion = FocalDiceloss()

    # === Resume from checkpoint ===
    if cfg.resume is not None:
        checkpoint = torch.load(cfg.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
        print(f"Resumed from checkpoint: {cfg.resume}")

    # === Data loaders ===
    train_ds = TrainingDataset(data_dir=Config["data_dir"])
    test_ds = TestingDataset(data_dir=Config["data_dir"])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, collate_fn=custom_collate_fn)

    log_dir = os.path.join(Config["out_dir"], "logs", f"{cfg.run_name}_{datetime.datetime.now():%Y%m%d-%H%M}.log")
    logger = get_logger(log_dir)
    logger.info(f"Configuration:\n{cfg}")

    best_score = 0
    train_len = len(train_loader)
    test_len = len(test_loader)

    for epoch in range(cfg.epochs):
        logger.info(f"\nEpoch {epoch+1}/{cfg.epochs}")
        epoch_start = time.time()
        os.makedirs(os.path.join(Config["out_dir"], "model_train", cfg.run_name), exist_ok=True)

        # === Train ===
        train_loss_list, train_metrics_sum = run_one_training_epoch(
            cfg, model, optimizer, train_loader, epoch, criterion, logger
        )
        scheduler.step()
        avg_loss = np.mean(train_loss_list)
        avg_train_metrics = [m / train_len for m in train_metrics_sum]
        train_metrics_dict = {cfg.metrics[i]: f"{avg_train_metrics[i]:.4f}" for i in range(len(cfg.metrics))}

        logger.info(f"Train Loss: {avg_loss:.4f} | Metrics: {train_metrics_dict} | LR: {scheduler.get_last_lr()[0]}")

        # === Evaluate ===
        test_metrics_sum = evaluate_one_epoch(cfg, model, test_loader, epoch)
        avg_test_metrics = [m / test_len for m in test_metrics_sum]
        test_metrics_dict = {cfg.metrics[i]: f"{avg_test_metrics[i]:.4f}" for i in range(len(cfg.metrics))}

        logger.info(f"Eval Metrics: {test_metrics_dict}")

        # === Save best model ===
        current_score = np.mean(avg_test_metrics)
        if current_score > best_score:
            best_score = current_score
            logger.info(f"New best at epoch {epoch+1}")
            save_path = os.path.join(Config["out_dir"], "model_train", cfg.run_name, f"epoch{epoch+1}_model.pth")
            torch.save({'model': model.float().state_dict(), 'optimizer': optimizer}, save_path)

        logger.info(f"Epoch time: {time.time() - epoch_start:.2f}s")


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)

    