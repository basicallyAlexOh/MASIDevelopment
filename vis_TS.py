

import os
import glob

from dataloader import train_dataloader
from monai.utils import set_determinism
from main import load_config

CONFIG_DIR = "/home/local/VANDERBILT/litz/github/MASILab/lobe_seg/configs"
config_TS = load_config(f"Config_0907_TS.YAML", CONFIG_DIR)
config_vlsp = load_config("Config_0416unet512peter.YAML", CONFIG_DIR)

images_TS = glob.glob(os.path.join(config_TS["data_dir"], "*.nii.gz"))[:5]
images_vlsp = glob.glob(os.path.join(config_vlsp["data_dir"], "*.nii.gz"))[:5]

TS_loader = train_dataloader(config_TS, images_TS)
vlsp_loader = train_dataloader(config_vlsp, images_vlsp)

batch_TS = next(iter(TS_loader))
batch_vlsp = next(iter(vlsp_loader))

data_TS, label_TS = batch_TS["image"], batch_TS["label"]
data_vlsp, label_vlsp = batch_vlsp["image"], batch_vlsp["image"]

print(data_TS.shape)
print(data_vlsp.shape)
# model_dir = os.path.join(config["model_dir"], config_id)
# device = torch.device("cuda:0")
# set_determinism(seed=config["random_seed"])
# random.seed(config["random_seed"])

print('done')