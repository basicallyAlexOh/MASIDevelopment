"""Post process lobe segmentations"""
import os
import sys
import numpy as np
from skimage import measure
from scipy import ndimage
import argparse
import SimpleITK as sitk
from tqdm import tqdm
import glob
from lungmask import mask
from utils import read_image

def get_largest_cc(img, connectivity=1):
    """
    :param binaryImg: binary 3d float array
    :param connectivity: https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
    """
    merged = np.zeros(img.shape)
    for i in range(5):
        label = i+1
        binary = np.where(img==label, 1, 0)
        labels = measure.label(binary, connectivity=connectivity)
        largest_cc = labels==np.argmax(np.bincount(labels.flat, weights=binary.flat))
        merged += largest_cc*label
    return merged

def get_lungmaks(raw_path):
    """
    segment lungmask using R231 from https://github.com/JoHof/lungmask
    """
    raw_sitk = sitk.ReadImage(raw_path)
    lungmask = mask.apply(raw_sitk) # PAL orientation
    lungmask = np.swapaxes(lungmask, 0,2) # SAL -> LAS
    return lungmask


def nearest_label_filling(img, cc):
    """
    Motivation: Since finding CCs is a reductionary operation, previously labeled voxels may loose labels. The nearest lable filling algorithm label voxels in the lung field that were labeled before cc reduction. implementation:
    1. find signed distance transform of each lobe, where more negative is inside the segmentation and more positive is outside
    2. subtract the binary pre-cc image from the binary post-cc image to find voxels that lost labels
    3. for each such voxel, assign it the label that corresponds to the smallest dt value across all lobes
    4. find cc of filled segmentation - this will remove voxels in the background that were labeled
    """
    dst_no_labels = np.zeros((5, *img.shape))
    no_label = np.where(img, 1, 0) - np.where(cc, 1, 0)
    for i in range(5):
        label = i+1
        binary = np.where(cc==label, 1, 0)
        inv_binary = np.where(cc==label, 0, 1)
        dst = -ndimage.distance_transform_cdt(binary) + ndimage.distance_transform_cdt(inv_binary)
        dst_no_labels[i, :,:,:] = np.where(no_label, dst, 0)

    nearest = np.argmin(dst_no_labels, axis=0)
    nearest = np.where(no_label, nearest + 1, 0)
    filled = cc + nearest
    return get_largest_cc(filled)

def lungmask_filling(cc, raw_path):
    lungmask = get_lungmaks(raw_path)
    dst_no_labels = np.zeros((5, *cc.shape))
    no_label = np.where(lungmask, 1, 0) - np.where(cc, 1, 0)
    for i in range(5):
        label = i+1
        binary = np.where(cc==label, 1, 0)
        inv_binary = np.where(cc==label, 0, 1)
        dst = -ndimage.distance_transform_cdt(binary) + ndimage.distance_transform_cdt(inv_binary)
        dst_no_labels[i, :,:,:] = np.where(no_label, dst, 0)

    nearest = np.argmin(dst_no_labels, axis=0)
    nearest = np.where(no_label, nearest + 1, 0)
    filled = cc + nearest
    return get_largest_cc(filled)


def postprocess_dir(seg_dir, out_dir):
    """Run post process for a directory"""
    # segs = glob.glob(os.path.join(seg_dir), "*.mhd")
    for seg in tqdm(os.listdir(seg_dir)):
    # for seg in tqdm(segs):
        seg_sitk = sitk.ReadImage(os.path.join(seg_dir, seg))
        seg_img = sitk.GetArrayFromImage(seg_sitk)
        ## cc -> nearest label filling -> cc
        postprocess = nearest_label_filling(seg_img, get_largest_cc(seg_img))

        postprocess_sitk = sitk.GetImageFromArray(postprocess)
        postprocess_sitk.CopyInformation(seg_sitk)
        sitk.WriteImage(postprocess_sitk, os.path.join(out_dir, os.path.basename(seg)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg-dir', type=str)
    parser.add_argument('--out-dir', type=str)
    args = parser.parse_args()
    postprocess_dir(args.seg_dir, args.out_dir)