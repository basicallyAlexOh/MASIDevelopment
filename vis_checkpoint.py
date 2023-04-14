"""Visualize model checkpoints and inferences"""
import os
import sys
from models import unet64, unet128, unet256, unet512
from main import load_config
import Experiment

if __name__ == "__main__":
    CONFIG_DIR = "/home/local/VANDERBILT/ohas/Desktop/Programming/new_lobe/lobe_seg/configs"
    config_id = sys.argv[1]
    model_name = sys.argv[2]
    config = load_config(f"Config_{config_id}.YAML", CONFIG_DIR)

    exp = Experiment.Experiment(config, config_id, unet128)
    exp.vis_model(model_name, nvals=50)
