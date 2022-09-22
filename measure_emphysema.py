"""Compute emphysema involvement by lobe"""

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
import numpy as np
from dataloader import infer_dataloader, npy_test_loader
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
)
from postprocess import get_largest_cc, lungmask_filling
sys.path.append("/home/local/VANDERBILT/litz/github/MASILab/thoraxtools/func")

lobe_label = ["LUL","LLL","RUL", "RML", "RLL", "Left", "Right", "All"]

def main(model, loader, device, out_path):
    model.eval()
    rows = []

    # append to existing file
    # old_df = pd.read_csv(out_path, dtype={'scanid':str})
    # done = old_df['scanid'].unique().tolist()

    with torch.no_grad():
        for batch in tqdm(loader):
            try:
                data, image_path = batch["image"].to(device), batch["image_path"][0]
                scanid = os.path.basename(image_path).split(".nii.gz")[0]
                raw_nii = nib.load(image_path)
                raw_img = raw_nii.get_fdata()
                axcodes = nib.orientations.aff2axcodes(raw_nii.affine)
                axcodes = ''.join(axcodes)
                sx, sy, sz = raw_nii.header.get_zooms()
                spatial_size = raw_nii.shape
                # adjust sizing for VRAM constraints
                # if spatial_size[0] > 600:
                #     sx, sy, sz = sx*1.5, sy*1.5, sz*1.5
                #     spatial_size = (round(spatial_size[0]/1.5), round(spatial_size[1]/1.5), round(spatial_size[2]/1.5))

                pred = sliding_window_inference(data, config["crop_shape"], 1, model)
                post_pred_transforms = Compose([
                    EnsureType(),
                    AsDiscrete(argmax=True),
                    Orientation(axcodes=axcodes),
                    Spacing(pixdim=(sx, sy, sz), mode="nearest"),
                    Resize(spatial_size=spatial_size, mode="nearest"),
                ])
                pred = post_pred_transforms(pred[0])
                label_map = pred[0].detach().cpu().numpy()
                label_map = get_largest_cc(label_map)

                # compute emphysema
                sx, sy, sz = raw_nii.header.get_zooms() #in mm
                voxelcc = sx*sy*sz/1000 # voxel size in cubic cm

                lobe_rows = []
                for i in range(5):
                    lobe_mask = np.where(label_map==i+1, 1, 0)
                    lobe_raw = np.multiply(raw_img, lobe_mask)
                    LAV = np.sum(np.where(lobe_raw < -950, 1, 0))*voxelcc
                    lobe_vol = np.sum(lobe_mask)*voxelcc
                    LAVp = LAV/lobe_vol
                    residual = lobe_vol - LAV
                    residualp = residual/lobe_vol

                    row = {"scanid":scanid, "lobe":lobe_label[i], "lobe_vol":lobe_vol, "LAV": LAV,
                        "LAVp": LAVp, "residual_vol": residual, "residualp": residualp}
                    lobe_rows.append(row)
                
                #  compute volume measures for left, right, all
                left = lobe_rows[:2]
                left_vol = sum([i["lobe_vol"] for i in left])
                left_LAV = sum([i["LAV"] for i in left])
                left_LAVp = left_LAV/left_vol
                left_residual = left_vol - left_LAV
                left_residualp = left_residual/left_vol
                left_row = [{"scanid":scanid, "lobe":lobe_label[5], "lobe_vol":left_vol, "LAV": left_LAV,
                        "LAVp": left_LAVp, "residual_vol": left_residual, "residualp": left_residualp}]

                right = lobe_rows[2:5]
                right_vol = sum([i["lobe_vol"] for i in right])
                right_LAV = sum([i["LAV"] for i in right])
                right_LAVp = right_LAV/right_vol
                right_residual = right_vol - right_LAV
                right_residualp = right_residual/right_vol
                right_row = [{"scanid":scanid, "lobe":lobe_label[6], "lobe_vol":right_vol, "LAV": right_LAV,
                        "LAVp": right_LAVp, "residual_vol": right_residual, "residualp": right_residualp}]

                all_vol = left_vol + right_vol
                all_LAV = left_LAV + right_LAV
                all_LAVp = all_LAV/all_vol
                all_residual = all_vol - all_LAV
                all_residualp = all_residual/all_vol
                all_row = [{"scanid":scanid, "lobe":lobe_label[7], "lobe_vol":all_vol, "LAV": all_LAV,
                        "LAVp": all_LAVp, "residual_vol": all_residual, "residualp": all_residualp}]
                
                rows = rows + lobe_rows + left_row + right_row + all_row
                df = pd.DataFrame(rows)
                df.to_csv(out_path, mode='a', header=False, index=False)
            except Exception as e:
                print(batch["image_path"][0])
                print(e)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-id', type=str)
    parser.add_argument('--out-name', type=str, default='/home/local/VANDERBILT/litz/github/MASILab/lobe_seg/test.csv')
    parser.add_argument('--sample', type=str, default='/home/local/VANDERBILT/litz/github/MASILab/lobe_seg/nlst_sample.csv')
    args = parser.parse_args()

    CONFIG_DIR = "/home/local/VANDERBILT/litz/github/MASILab/lobe_seg/configs"
    config_id = args.config_id
    config = load_config(f"Config_{config_id}.YAML", CONFIG_DIR)

    data_dir = config["data_dir"]
    model_dir = os.path.join(config["model_dir"], config_id)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(model_dir, args.out_name)
    model_path = config["pretrained"]

    set_determinism(seed=config["random_seed"])
    random.seed(config["random_seed"])
    device = torch.device(config["device"])

    qa_fail = [
        "/home/litz/data/NLST/T0_softkernel/1.2.840.113654.2.55.335457016568280897088797156541193659691.nii.gz", 
        "/home/litz/data/NLST/T0_softkernel/1.2.840.113654.2.55.41002169116289661221185422515590944596.nii.gz",
        "/home/litz/data/NLST/T0_softkernel/1.2.840.113654.2.55.225791384554099674765444733605720026767.nii.gz"
    ]
    if args.sample:
        df = pd.read_csv(args.sample)
        # skip scans that have already been analyzed
        if os.path.exists(out_path):
            old_df = pd.read_csv(out_path, dtype={'scanid':str})
            done = old_df.iloc[:,1].unique().tolist()

            images = [os.path.join(config["data_dir"], f"{scanid}.nii.gz") for scanid in df['series_uid'].tolist() if scanid not in done]
        else:
            images = [os.path.join(config["data_dir"], f"{scanid}.nii.gz") for scanid in df['series_uid'].tolist()]
        # images with faulty headers
        images = [i for i in images if i not in qa_fail]
    else: 
        images = glob.glob(os.path.join(config["data_dir"], "*.nii.gz"))
    loader = infer_dataloader(config, images)

    model = unet512(6).to(device)
    model.load_state_dict(torch.load(model_path))

    main(model, loader, device, out_path)