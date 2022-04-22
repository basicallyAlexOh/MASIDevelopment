"""Code for visualizing volumes and results using montages"""

import os
import sys
import argparse
import SimpleITK as sitk
sys.path.append("/home/local/VANDERBILT/litz/github/MASILab/thoraxtools/func")
import vis.paral_clip_overlay_mask as paral_clip

def clip_raw(img_dir, out_dir, planes=(1,1,1), x=4, y=4):
    """
    Clip volume into a x by y montage for specified planes. For visualizing volumes without mask overlay
    :param planes: which planes to generate montages for. (axial, coronal, sagital)
    """
    for img in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img)
        img_name = img.split(".")[0]
        filetype = img.split(".")[1]
        img_sitk = sitk.ReadImage(img_path)
        img_img = sitk.GetArrayFromImage(img_sitk)
        if planes[0]:
            paral_clip.mutliple_clip_overlay_from_list(img_img, os.path.join(out_dir, f"{img_name}_axial.{filetype}"), clip_plane='axial', dim_x=x, dim_y=y)
        if planes[1]:
            paral_clip.mutliple_clip_overlay_from_list(img_img, os.path.join(out_dir, f"{img_name}_coronal.{filetype}"), clip_plane='coronal', dim_x=x, dim_y=y)
        if planes[2]:
            paral_clip.mutliple_clip_overlay_from_list(img_img, os.path.join(out_dir, f"{img_name}_sagittal.{filetype}"), clip_plane='sagittal', dim_x=x, dim_y=y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str)
    parser.add_argument('--out-dir', type=str)
    parser.add_argument('--axial', action='store_true', default=False)
    parser.add_argument('--coronal', action='store_true', default=False)
    parser.add_argument('--sagittal', action='store_true', default=False)
    args = parser.parse_args()

    clip_raw(args.img_dir, args.out_dir, planes=(args.axial, args.coronal, args.sagittal))
