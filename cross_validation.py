"""
5-fold cross validation:
Run separate train and test on all fold combinations
"""
import argparse
from monai.utils import set_determinism

from monai.metrics import DiceMetric
from monai.losses import DiceCELoss

from torch.utils.tensorboard import SummaryWriter
import torch
import os
import random
from pathlib import Path
import MetricLogger
from dataloader import train_dataloader, val_dataloader, test_dataloader
from models import unet256, unet512, unet1024
from train import train
from test import test
from luna16_preprocess import get_kfolds
from main import load_config
from scheduler import WarmupCosineSchedule

def train_one_fold(config, config_id, k):
    k = int(k)
    print(f"Training on all folds except {k}")
    # unwrap directory paths
    MODEL_DIR = os.path.join(config["model_dir"], config_id, f"fold{k}")
    CHECKPOINT_DIR = os.path.join(config["checkpoint_dir"], config_id, f"fold{k}")
    LOG_DIR = os.path.join(config["log_dir"], config_id, f"fold{k}")

    # Set randomness
    set_determinism(seed=config["random_seed"])
    random.seed(config["random_seed"])

    # Make paths
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    # Logger
    logger = MetricLogger.MetricLogger(config, config_id)
    writer = SummaryWriter(log_dir=LOG_DIR)

    # Load data
    folds = get_kfolds(config["kfolds_path"])
    images = [x for i, x in enumerate(folds) if i!=(k-1)] # all folds except k
    images = [x for fold in images for x in fold] # flatten list of lists
    val_size = int(len(images)*0.2)
    random.shuffle(images)
    val_images, train_images  = images[:val_size], images[val_size:]
    # get dataloaders
    train_loader = train_dataloader(config, train_images)
    val_loader = val_dataloader(config, val_images)

    # Initialize Model, Loss, and Optimizer
    device = torch.device("cuda:0")

    if config["model"] == 'unet512':
        model = unet512(6).to(device)
    elif config["model"] == 'unet1024':
        model = unet1024(6).to(device)
    elif config["model"] == 'unetr16':
        model = unetr16(6).to(device)
    else:
        model = unet256(6).to(device)
    # loss_function = DiceLoss(include_background=config["include_bg_loss"], to_onehot_y=True, softmax=True)
    loss_function = DiceCELoss(include_background=config["include_bg_loss"], to_onehot_y=True, softmax=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    dice_metric = DiceMetric(False, reduction="mean", get_not_nans=False)
    start_epoch = 0

    # scheduler
    n_batches = len(train_loader)
    print(f"Total steps: {config['epochs']*n_batches}")
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=config["warmup_steps"], 
        t_total=config["epochs"]*n_batches, last_epoch=start_epoch*n_batches-1)

    # Finetune pretrained model if indicated
    if config["pretrained"]:
        print(f"Fine tuning model from {config['pretrained']}")
        pretrained = torch.load(config['pretrained'])
        model.load_state_dict(pretrained)

    train(config,
          config_id,
          model,
          device,
          optimizer,
          scheduler,
          loss_function,
          dice_metric,
          train_loader,
          val_loader,
          (start_epoch, config["epochs"]),
          logger,
          writer,
          CHECKPOINT_DIR,
          MODEL_DIR)

def test_one_fold(config, config_id, out_name, k, output_seg=True, output_clip=True):
    k = int(k)
    MODEL_DIR = os.path.join(config["model_dir"], config_id, f"fold{k}")
    metrics_path = os.path.join(MODEL_DIR, out_name)
    seg_dir = os.path.join(MODEL_DIR, 'segs') if output_seg else False
    clip_dir = os.path.join(MODEL_DIR, 'clips') if output_clip else False
    model_path = os.path.join(MODEL_DIR, f"{config_id}_best_model.pth")

    if output_seg:
        Path(seg_dir).mkdir(parents=True, exist_ok=True)
    if output_clip:
        Path(clip_dir).mkdir(parents=True, exist_ok=True)
        
    # Set randomness
    set_determinism(seed=config["random_seed"])
    random.seed(config["random_seed"])

    # Load data
    images = get_kfolds(config["kfolds_path"])
    test_images = images[k-1]
    test_loader = test_dataloader(config, test_images)

    # Initialize Model and test metric
    device = torch.device("cuda:0")
    if config["model"] == 'unet512':
        model = unet512(6).to(device)
    elif config["model"] == 'unet1024':
        model = unet1024(6).to(device)
    else:
        model = unet256(6).to(device)
    # Set metric to compute average over each class
    test_metric = DiceMetric(include_background=False, reduction="none")

    test(config,
         config_id,
         device,
         model,
         model_path,
         test_metric,
         test_loader,
         metrics_path,
        seg_dir,
        clip_dir)

if __name__ == "__main__":
    # python3 cross_validation.py --config-id 0418cv_luna16 --train --test
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-id', type=str)
    parser.add_argument('--out-name', type=str, default='test.csv')
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--output-seg', action='store_true', default=False)
    parser.add_argument('--output-clip', action='store_true', default=False)
    args = parser.parse_args()

    CONFIG_DIR = "/home/local/VANDERBILT/litz/github/MASILab/lobe_seg/configs"
    config = load_config(f"Config_{args.config_id}.YAML", CONFIG_DIR)

    if args.train:
        train_one_fold(config, args.config_id, args.k)
    if args.test:
        test_one_fold(config, args.config_id, args.out_name, args.k, output_seg=args.output_seg, output_clip=args.output_clip)
