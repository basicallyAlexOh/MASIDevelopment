"""Code for visualizing volumes and results using montages"""

import os
import sys
import argparse
import SimpleITK as sitk
import nibabel as nib
from tqdm import tqdm
sys.path.append("/home/local/VANDERBILT/litz/github/MASILab/thoraxtools/func")
import vis.paral_clip_overlay_mask as paral_clip
from utils import parse_id, sitk2np

def clip_raw(img_dir, out_dir, planes=(1,1,1), x=4, y=4):
    """
    Clip volume into a x by y montage for specified planes. For visualizing volumes without mask overlay
    :param planes: which planes to generate montages for. (axial, coronal, sagital)
    """
    for img in tqdm(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img)
        img_name = img.split(".")[0]
        img_sitk = sitk.ReadImage(img_path)
        img_img = sitk.GetArrayFromImage(img_sitk)
        if planes[0]:
            paral_clip.multiple_clip_overlay_from_np_sitk(img_img, os.path.join(out_dir, f"{img_name}_axial.png"), clip_plane='axial', dim_x=x, dim_y=y)
        if planes[1]:
            paral_clip.multiple_clip_overlay_from_np_sitk(img_img, os.path.join(out_dir, f"{img_name}_coronal.png"), clip_plane='coronal', dim_x=x, dim_y=y)
        if planes[2]:
            paral_clip.multiple_clip_overlay_from_np_sitk(img_img, os.path.join(out_dir, f"{img_name}_sagittal.png"), clip_plane='sagittal', dim_x=x, dim_y=y)

def clip_seg(raw_dir, seg_dir, out_dir, planes=(1,1,1), x=4, y=4, io="sitk", suffix=".nii.gz"):
    for seg_name in os.listdir(seg_dir):
        scanid = parse_id(seg_name)
        suffix = os.path.splitext(seg_name)[1]
        seg_path = os.path.join(seg_dir, seg_name)
        raw_path = os.path.join(raw_dir, f"{scanid}.nii.gz")
        # raw_path = os.path.join(raw_dir, f"{scanid}.nii.gz") if suffix=='.gz' else os.path.join(raw_dir, f"{scanid}.mhd")
        # label_path = os.path.join(label_dir, f"lvlsetseg_{scanid}{suffix}")

        if io=="sitk":
            seg = sitk2np(seg_path)
            raw = sitk2np(raw_path)

        if io=="nib":
            seg = nib.load(seg_path).get_fdata()
            raw = nib.load(raw_path).get_fdata()

        if planes[0]:
            paral_clip.multiple_clip_overlay_with_mask_from_npy(raw, seg, \
                os.path.join(out_dir, f"{scanid}_seg_axial.png"), 'axial', dim_x=x, dim_y=y)
        if planes[1]:
            paral_clip.multiple_clip_overlay_with_mask_from_npy(raw, seg, \
                os.path.join(out_dir, f"{scanid}_seg_coronal.png"), 'coronal', dim_x=x, dim_y=y)
        if planes[2]:
            paral_clip.multiple_clip_overlay_with_mask_from_npy(raw, seg, \
                os.path.join(out_dir, f"{scanid}_seg_sagittal.png"), 'sagittal', dim_x=x, dim_y=y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str)
    parser.add_argument('--out-dir', type=str)
    parser.add_argument('--seg-dir', type=str)
    parser.add_argument('--clip-mask', action='store_true', default=False)
    parser.add_argument('--axial', action='store_true', default=False)
    parser.add_argument('--coronal', action='store_true', default=False)
    parser.add_argument('--sagittal', action='store_true', default=False)
    args = parser.parse_args()

    if args.clip_mask:
        clip_seg(args.img_dir, args.seg_dir, args.out_dir, planes=(args.axial, args.coronal, args.sagittal))
    else:
        clip_raw(args.img_dir, args.out_dir, planes=(args.axial, args.coronal, args.sagittal))
