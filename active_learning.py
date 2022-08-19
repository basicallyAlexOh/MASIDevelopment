"""Utility for doing active learning"""

import os
import sys
import re
import shutil
import random
import pathlib
import SimpleITK as sitk
from postprocess import lungmask_filling, get_largest_cc
sys.path.append("/home/local/VANDERBILT/litz/github/MASILab/thoraxtools/func")
import vis.paral_clip_overlay_mask as overlay
import pandas as pd
from skimage.transform import resize
from tqdm import tqdm
import nibabel as nib
import math

def sample_candidates(img_dir, dst_dir):
    """Random sample of VLSP cohort"""
    random.seed(0)
    sample = random.sample(os.listdir(img_dir), 100)
    for img_name in sample:
        src = os.path.join(img_dir, img_name)
        dst = os.path.join(dst_dir, img_name)
        os.symlink(src, dst)

def clip_al_candidates(cand_path, mask_dir, out_dir):
    cand_df = pd.read_csv(cand_path)

    for path in cand_df["input_path"]:
        mask_path = os.path.join(mask_dir, f"lvlsetseg_{os.path.basename(path)}")
        overlay.multiple_clip_overlay_with_mask(path, mask_path, os.path.join(out_dir, f"{os.path.basename(path)}_axial.png"), \
                                                clip_plane='axial')
        overlay.multiple_clip_overlay_with_mask(path, mask_path,
                                                os.path.join(out_dir, f"{os.path.basename(path)}_coronal.png"), \
                                                clip_plane='coronal')
        overlay.multiple_clip_overlay_with_mask(path, mask_path,
                                                os.path.join(out_dir, f"{os.path.basename(path)}_sagittal.png"), \
                                                clip_plane='sagittal')

def resize_label(label_path, img_path):
    """resize label to image shape"""
    label_sitk = sitk.ReadImage(label_path)
    label = sitk.GetArrayFromImage(label_sitk)

    # low memory read of img
    reader = sitk.ImageFileReader()
    reader.SetFileName(img_path)
    reader.ReadImageInformation()
    s = reader.GetSize()
    s = (s[2], s[0], s[1])

    resized = resize(label, s, order=0) # nearest neighbor resize

    resized_sitk = sitk.GetImageFromArray(resized)
    resized_sitk.SetOrigin(reader.GetOrigin())
    resized_sitk.SetSpacing(reader.GetSpacing())
    resized_sitk.SetDirection(reader.GetDirection())
    return resized_sitk

def copy_al_candidates(cand_path, img_dir, label_dir, dst_dir):
    """ copy labels for active learning and resize them to image space"""
    # cand_df = pd.read_csv(cand_path)
    # for path in tqdm(cand_df["input_path"]):
    for path in tqdm(os.listdir(dst_dir)):
        # src_path = os.path.join(src_dir, f"lvlsetseg_{os.path.basename(path)}")
        # dst_path = os.path.join(dst_dir, f"{os.path.basename(path).split('.')[0]}_lvlsetseg.nii.gz")
        # shutil.copy(src_path, dst_dir)
        img_path = os.path.join(img_dir, path)
        label_path = os.path.join(label_dir, f"lvlsetseg_{path}")
        # print(label_path)
        resized = resize_label(label_path, img_path)
        dst_path = os.path.join(dst_dir, f"{path.split('.')[0]}_lvlsetseg.nii.gz")
        sitk.WriteImage(resized, dst_path)

def copy_al_train(label_dir, luna_dir, vlsp_dir, dst_dir):
    """Copy images for training. Convert all to nifti"""
    for path in tqdm(os.listdir(label_dir)):
        name = os.path.splitext(path)[0]
        suffix = os.path.splitext(path)[1]
        # src = os.path.join(luna_dir)
        if suffix == '.nrrd':
            fname = f"{name[:-17]}.mhd"
            src = os.path.join(luna_dir, fname)
            if os.path.isfile(src):
                img_sitk = sitk.ReadImage(src)
                sitk.WriteImage(img_sitk, os.path.join(dst_dir, f"{os.path.splitext(fname)[0]}.nii.gz"))
        else:
            fname = f"{name[:-14]}.nii.gz"
            src = os.path.join(vlsp_dir, fname)
            shutil.copyfile(src, os.path.join(dst_dir, fname))
            # os.symlink(src, os.path.join(dst_dir, fname))

        # print(src)

        # os.symlink(src, os.path.join(dst_dir, fname))

def convert_nifti(src_dir, dst_dir):
    """convert directory to nii.gz files"""
    for name in tqdm(os.listdir(src_dir)):
        if os.path.splitext(name)[1] == ".nrrd":
            img_sitk = sitk.ReadImage(os.path.join(src_dir, name))
            sitk.WriteImage(img_sitk, os.path.join(dst_dir, f"{os.path.splitext(name)[0]}.nii.gz"))

def preprocess_labels(src_dir, dst_dir, raw_dir):
    """Preprocess al labels with: CC and lungmask filling"""
    for fname in tqdm(os.listdir(src_dir)):
        # skip if already preprocessed
        if os.path.exists(os.path.join(dst_dir, fname)):
            continue
        nii = nib.load(os.path.join(src_dir, fname))
        # skip large images
        if math.prod(nii.shape) >= 768*768*500:
            continue
        img = nii.get_fdata()
        scanid = fname.split("_")[0]
        raw_path = os.path.join(raw_dir, f"{scanid}.nii.gz")
        preproc_img = lungmask_filling(get_largest_cc(img), raw_path)
        
        preproc_nii = nib.Nifti1Image(preproc_img, header=nii.header, affine=nii.affine)
        nib.save(preproc_nii, os.path.join(dst_dir, fname))

def clip_seg(seg_dir, clip_dir, raw_dir):
    for fname in tqdm(os.listdir(seg_dir)):
        scanid = fname.split("_")[0]
        seg_path = os.path.join(seg_dir, fname)
        raw_path = os.path.join(raw_dir, f"{scanid}.nii.gz")
        overlay.multiple_clip_overlay_with_mask(raw_path, seg_path,
            os.path.join(clip_dir, f"{scanid}_coronal.png"),
            clip_plane='coronal',
            img_vrange=(-1000, 0))

if __name__ == "__main__":
    args = sys.argv[1:]
    # sample_candidates(*args)
    # clip_al_candidates(*args)
    # copy_al_candidates(*args)
    # copy_al_train(*args)
    # convert_nifti(*args)
    preprocess_labels(*args)
    # clip_seg(*args)