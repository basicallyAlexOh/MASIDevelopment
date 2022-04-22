"""Test models trained with Peter's scripts"""
from monai.utils import set_determinism

from monai.metrics import DiceMetric
import torch
import os
import sys
import yaml
import random
import glob
from dataloader import val_dataloader
from models import unet256, unet512, unet1024
from test import test

def run_test(config, config_id, out_name):
    DATA_DIR = config["test_dir"]
    out_path = os.path.join(config["model_dir"], out_name)
    model_path = os.path.join(config["model_dir"], "best_metric_model.pth")
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
    # Validation/Test
    CONFIG_DIR = "/home/local/VANDERBILT/litz/github/MASILab/lobe_seg/configs"
    config_id, out_name = sys.argv[1], sys.argv[2]
    config = load_config(f"Config_{config_id}.YAML", CONFIG_DIR)
    run_test(config, config_id, out_name)
