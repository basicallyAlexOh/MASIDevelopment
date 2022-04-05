"""Visualize model checkpoints and inferences"""
import os
import sys
from models import unet256, unet512
from main import load_config
import Experiment

if __name__ == "__main__":
    CONFIG_DIR = "/home/local/VANDERBILT/litz/github/MASILab/lobe_seg/configs"
    config_id = sys.argv[1]
    checkpoint_name = sys.argv[2]
    config = load_config(f"Config_{config_id}.YAML", CONFIG_DIR)

    exp = Experiment.Experiment(config, config_id, unet512)
    exp.vis_checkpoint(checkpoint_name)