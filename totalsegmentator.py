"""Scripts for testing TotalSegmentator https://github.com/wasserth/TotalSegmentator"""

import os
import sys
import shutil
from pathlib import Path
import glob

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
    


if __name__ == "__main__":
    args = sys.argv[1:]
    copy_dataset(*args) 




