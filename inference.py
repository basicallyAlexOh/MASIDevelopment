"""Acquire and display lobe segmentation inferences"""

import os
import sys
import glob
from tqdm import tqdm
from main import load_config
import random
from pathlib import Path
import argparse
import pandas as pd
import math

from dataloader import infer_dataloader, test_dataloader
from models import unet512
import torch
import nibabel as nib
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    AsDiscrete,
    EnsureType,
    AsDiscrete,
    Spacing,
    Resize,
    Orientation,
    AddChannel
)
from postprocess import get_largest_cc, lungmask_filling
sys.path.append("/home/local/VANDERBILT/litz/github/MASILab/thoraxtools/func")
import vis.paral_clip_overlay_mask as overlay


def infer(device, model, infer_loader, seg_dir, clip_dir=None):
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(infer_loader):
            data, image_path = batch["image"].to(device), batch["image_path"][0]
            fname = os.path.basename(image_path).split(".")[0]
            
            #  check if segmentation has been done
            if os.path.exists(os.path.join(seg_dir, f"{fname}.nii.gz")):
                continue 

            raw_nii = nib.load(image_path)
            axcodes = nib.orientations.aff2axcodes(raw_nii.affine)
            axcodes = ''.join(axcodes)
            pixdim = raw_nii.header.get_zooms()
            spatial_size = raw_nii.shape
            print(spatial_size)
            # skip if volume exceeds VRAM constraints
            if math.prod(spatial_size) > 512*512*500:
                continue
            post_pred_transforms = Compose([
                EnsureType(),
                AsDiscrete(argmax=True),
                Orientation(axcodes=axcodes),
                Spacing(pixdim=pixdim, mode="nearest"), 
                Resize(spatial_size=spatial_size, mode="nearest"),
            ])
            pred = sliding_window_inference(data, config["crop_shape"], 4, model)
            pred = post_pred_transforms(pred[0])
            label_map = pred[0].detach().cpu().numpy()
            label_map = lungmask_filling(get_largest_cc(label_map), image_path)

            
            label_map_nii = nib.Nifti1Image(label_map, header=raw_nii.header, affine =raw_nii.affine)
            nib.save(label_map_nii, os.path.join(seg_dir, f"{fname}.nii.gz"))
            vis([image_path], seg_dir, clip_dir)

            # resize raw and visualize overlay
            raw_transforms = Compose([EnsureType(), AddChannel(), Resize(spatial_size=spatial_size, mode="trilinear")])
            raw_img = raw_nii.get_fdata()
            raw_img = raw_transforms(raw_img)[0]
            overlay.multiple_clip_overlay_with_mask_from_npy(raw_img, label_map,
                os.path.join(clip_dir, f"{fname}_coronal.png"), 
                clip_plane="coronal", 
                img_vrange=(-1000,0))

def vis(images, seg_dir, clip_dir):
    for image_path in tqdm(images):
        fname = os.path.basename(image_path)
        seg_path = os.path.join(seg_dir, fname)
        overlay.multiple_clip_overlay_with_mask(image_path, seg_path,
            os.path.join(clip_dir, f"{fname.split('.')[0]}_coronal.png"),
            clip_plane='coronal',
            img_vrange=(-1000, 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer', action='store_true', default=False)
    parser.add_argument('--vis', action='store_true', default=False)
    args = parser.parse_args()

    # Setup
    CONFIG_DIR = "/home/local/VANDERBILT/litz/github/MASILab/lobe_seg/configs"
    config_id = "infer_vlsp"
    config = load_config(f"Config_{config_id}.YAML", CONFIG_DIR)

    data_dir = config["data_dir"]
    model_dir = os.path.join(config["model_dir"])
    model_path = config["pretrained"]
    clip_dir = os.path.join(config["clip_dir"])
    seg_dir = os.path.join(config["seg_dir"])
    Path(clip_dir).mkdir(parents=True, exist_ok=True)
    Path(seg_dir).mkdir(parents=True, exist_ok=True)

    set_determinism(seed=config["random_seed"])
    random.seed(config["random_seed"])
    device = torch.device(config["device"])

    # Load N random images
    images = glob.glob(os.path.join(data_dir, config["image_type"]))
    if config["sample_size"]:
        images = random.sample(images, config["sample_size"])
    infer_loader = infer_dataloader(config, images)

    # load model
    model = unet512(6).to(device)
    model.load_state_dict(torch.load(model_path))

    # csv for qualitatively grading inferences (sensitivity analysis)
    grade_csv_path = "/home/local/VANDERBILT/litz/github/MASILab/lobe_seg/infer_vlsp.csv"
    scanids = [os.path.basename(img).split('.')[0] for img in images]
    grade_df = pd.DataFrame({"scanid":sorted(scanids)})
    grade_df.to_csv(grade_csv_path, index_label=False, index=False)

    if args.infer:
        infer(device, model, infer_loader, seg_dir, clip_dir)
    if args.vis:
        vis(images, seg_dir, clip_dir)

    
