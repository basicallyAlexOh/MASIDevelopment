"""Lobe segmentation with MONAI"""

from monai.utils import set_determinism

from monai.metrics import DiceMetric
from monai.losses import DiceLoss

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import torch
import os
import sys
import yaml
import random
import glob
from pathlib import Path
import MetricLogger
from dataloader import train_dataloader, val_dataloader
from models import unet256, unet512, unet1024
from train import train
from test import test

def run_train(config, config_id):
    # unwrap directory paths
    MODEL_DIR = os.path.join(config["model_dir"], config_id)
    CHECKPOINT_DIR = os.path.join(config["checkpoint_dir"], config_id)
    LOG_DIR = os.path.join(config["log_dir"], config_id)
    DATA_DIR = config["data_dir"]

    # Set randomness
    set_determinism(seed=config["random_seed"])
    random.seed(config["random_seed"])

    # Make paths
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    # Logger
    logger = MetricLogger.MetricLogger(config, config_id)
    writer = SummaryWriter(log_dir=LOG_DIR)

    # Load data
    images = sorted(glob.glob(os.path.join(DATA_DIR, config["image_type"])))
    # limit sample size if specified
    if config["sample_size"]:
        images = random.sample(images, config["sample_size"])
    # split dataset into train and validation
    val_size = int(len(images) * config["val_ratio"])
    random.shuffle(images)
    val_images, train_images = images[:val_size], images[val_size:]
    # get dataloaders
    train_loader = train_dataloader(config, train_images)
    val_loader = val_dataloader(config, val_images)

    # LABEL_SHAPE = (512, 512, 320)  # All labels have this shape, but input shapes vary

    # Initialize Model, Loss, and Optimizer
    device = torch.device("cuda:0")

    if config["model"] == 'unet512':
        model = unet512(6).to(device)
    elif config["model"] == 'unet1024':
        model = unet1024(6).to(device)
    else:
        model = unet256(6).to(device)
    loss_function = DiceLoss(include_background=config["include_bg_loss"], to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    start_epoch = 0

    # Resume training from checkpoint if indicated
    if config["checkpoint"]:
        print(f"Resuming training of {config_id} from {config['checkpoint']}")
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, config["checkpoint"]))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

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
          loss_function,
          dice_metric,
          train_loader,
          val_loader,
          (start_epoch, config["epochs"]),
          logger,
          writer,
          CHECKPOINT_DIR,
          MODEL_DIR)

def run_test(config, config_id, out_name):
    DATA_DIR = config["test_dir"]
    MODEL_DIR = os.path.join(config["model_dir"], config_id)
    out_path = os.path.join(MODEL_DIR, out_name)
    model_path = os.path.join(MODEL_DIR, f"{config_id}_best_model.pth")

    # Set randomness
    set_determinism(seed=config["random_seed"])
    random.seed(config["random_seed"])

    # Load data
    images = sorted(glob.glob(os.path.join(DATA_DIR, config["image_type"])))
    test_loader = val_dataloader(config, images)

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
         out_path)

def load_config(config_name, config_dir):
    with open(os.path.join(config_dir, config_name)) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

if __name__ == "__main__":
    # Training
    CONFIG_DIR = "/home/local/VANDERBILT/litz/github/MASILab/lobe_seg/configs"
    config_id = sys.argv[1]
    config = load_config(f"Config_{config_id}.YAML", CONFIG_DIR)
    run_train(config, config_id)

    # Validation/Test
    # CONFIG_DIR = "/home/local/VANDERBILT/litz/github/MASILab/lobe_seg/configs"
    # config_id, out_name = sys.argv[1], sys.argv[2]
    # config = load_config(f"Config_{config_id}.YAML", CONFIG_DIR)
    # run_test(config, config_id, out_name)
