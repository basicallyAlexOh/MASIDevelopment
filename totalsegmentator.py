"""Scripts for testing TotalSegmentator https://github.com/wasserth/TotalSegmentator"""

import os
import sys
import shutil
from pathlib import Path
import glob
sys.path.append("/home/local/VANDERBILT/litz/github/MASILab/thoraxtools/func")
import vis.paral_clip_overlay_mask as overlay

def copy_dataset(src_dir, dst_dir):
    src_paths = glob.glob(os.path.join(src_dir, "*.nii.gz"))
    for src_path in src_paths:
        # create dir
        scanid = os.path.basename(src_path).split(".nii.gz")[0]
        dst_scan_dir = os.path.join(dst_dir, scanid)
        Path(dst_scan_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(dst_scan_dir, "segmentations")).mkdir(parents=True, exist_ok=True)
        # copy nifti into dirs
        dst_path = os.path.join(dst_scan_dir, "ct.nii.gz")
        shutil.copyfile(src_path, dst_path)
    
def vis_dataset(root_dir, label_dir, clip_dir):
    root_paths = glob.glob(os.path.join(root_dir, "*.nii.gz"))
    label_paths = [os.path.join(label_dir, os.path.basename(p)) for p in root_paths]
    for root_path, label_path in zip(root_paths, label_paths):
        scanid = os.path.basename(root_path).split(".nii.gz")[0]
        overlay.multiple_clip_overlay_with_mask(root_path, label_path, os.path.join(clip_dir, f"{scanid}.png"), clip_plane='coronal')

def copy_luna16(src_dir, dst_dir):
    """create symlinks of luna16 datset only"""
    src_paths = glob.glob(os.path.join(src_dir, "1.3.6.1.*"))
    dst_paths = [os.path.join(dst_dir, os.path.basename(p)) for p in src_paths]
    for src, dst in zip(src_paths, dst_paths):
        shutil.copyfile(src, dst)

if __name__ == "__main__":
    args = sys.argv[1:]
    # copy_dataset(*args) 
    # vis_dataset(*args)
    copy_luna16(
        "/home/litz/data/TotalSegmentator/model/dataset_rand_merged", 
        "/home/local/VANDERBILT/litz/data/luna16/TotalSegmentator")




