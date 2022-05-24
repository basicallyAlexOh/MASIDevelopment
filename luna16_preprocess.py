"""Data preprocess pipeline for LUNA16 dataset and its annotations from https://github.com/deep-voxel/automatic_pulmonary_lobe_segmentation_using_deep_learning"""
import os
import glob
import re
import shutil
import sys
import random
import SimpleITK as sitk
import numpy as np
from scipy import ndimage as ndi
import skimage.measure
from tqdm import tqdm
import csv
sys.path.append("/home/local/VANDERBILT/litz/github/MASILab/thoraxtools/func")
import vis.paral_clip_overlay_mask as overlay

def copy_dataset(raw_path, label_path, out_dir):
    """copy over images that we have labels for"""
    label_paths = glob.glob(os.path.join(label_path, "*.nrrd"))
    raw_names = [
        f"{re.match('^.*(?=_LobeSegmentation)', os.path.basename(path)).group()}.raw"
        for path in label_paths
    ]
    for raw_name in tqdm(raw_names):
        src = os.path.join(raw_path, raw_name)
        dst = os.path.join(out_dir, raw_name)
        shutil.copyfile(src, dst)

def train_test_split(root_dir, train_dir, test_dir):
    """Split dataset into train and test sets, and store in separate dirs"""
    scans = glob.glob(os.path.join(root_dir, "*.mhd"))
    scanids = [os.path.basename(scanid)[:-4] for scanid in scans]

    random.seed(1)
    random.shuffle(scanids)
    test_size = int(len(scanids) *0.3)
    test_images, train_images = scanids[:test_size], scanids[test_size:]
    for scanid in tqdm(train_images):
        # file paths of .mhd and .raw
        ids = glob.glob(os.path.join(root_dir, f"{scanid}.*"))
        for id in ids:
            shutil.copy(id, train_dir)
    for scanid in tqdm(test_images):
        # file paths of .mhd and .raw
        ids = glob.glob(os.path.join(root_dir, f"{scanid}.*"))
        for id in ids:
            shutil.copy(id, test_dir)

def generate_kfolds(root_dir, out_path, k=5):
    """split dataset into k folds and write to file"""
    scans = [os.path.join(root_dir, i) for i in os.listdir(root_dir) if os.path.splitext(i)[1] != ".raw"]
    # scans = glob.glob(os.path.join(root_dir, "*.mhd"))
    random.seed(2)
    random.shuffle(scans)
    test_size = int(len(scans)/k)
    folds = []
    for i in range(k):
        if i==k-1:
            ifold = scans[(i*test_size):]
        else:
            ifold = scans[(i*test_size):((i+1)*test_size)]
        folds.append(ifold)
    with open(out_path, "w") as f:
        wr = csv.writer(f)
        wr.writerows(folds)

def get_kfolds(kfolds_path, k=5):
    """
    read kfolds file and return list of lists
    :return: [[1st fold],..,[kth fold]]
    """
    folds = []
    with open(kfolds_path, "r") as f:
        for row in f:
            fold = row.rstrip().split(",")
            folds.append(fold)
    return folds

def fix_labels(label_dir, out_dir):
    """
    Limit labels to first n_classes and ensure classes are the following:
    0: background
    1: upper left
    2: lower left
    3: upper right
    4: middle right
    5: lower right
    Tang et al. (https://github.com/deep-voxel/automatic_pulmonary_lobe_segmentation_using_deep_learning) annotations are encoded as the following:
    0: background
    4: upper right
    5: middle right
    6: lower right
    7: upper left
    8: lower left
    """
    for label_path in tqdm(glob.glob(os.path.join(label_dir, "*nrrd"))):
        label_sitk = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label_sitk)
        
        # Any voxel with labels outside of 0 to (n_classes-1) will be set to 0.
        fix_label = np.where(label  > 8, 0, label)
        #
        fix_label = np.where(fix_label == 7, 1, fix_label)
        fix_label = np.where(fix_label == 8, 2, fix_label)
        fix_label = np.where(fix_label == 4, 3, fix_label)
        fix_label = np.where(fix_label == 5, 4, fix_label)
        fix_label = np.where(fix_label == 6, 5, fix_label)
        
        fix_label_sitk = sitk.GetImageFromArray(fix_label)
        fix_label_sitk.CopyInformation(label_sitk)
        sitk.WriteImage(fix_label_sitk, os.path.join(out_dir, os.path.basename(label_path)))

def generate_clips(raw_dir, out_dir):
    for raw_path in tqdm(glob.glob(os.path.join(raw_dir, "*.mhd"))):
        raw = sitk.ReadImage(raw_path)
        raw_img = sitk.GetArrayFromImage(raw)
        out_path = os.path.join(out_dir, f"{os.path.basename(raw_path)[:-4]}_coronal.png")
        overlay.multiple_clip_overlay_from_np_sitk(raw_img, out_path, clip_plane='coronal')

def apply_body_mask_dir(mask_dir, raw_dir, out_dir, filetype, bg_value=-1500):
    """ Uses existing body mask to set background pixels to some HU value"""
    for raw_path in tqdm(glob.glob(os.path.join(raw_dir, f"*.{filetype}"))):
        mask_path = os.path.join(mask_dir, os.path.basename(raw_path))
        raw = sitk.ReadImage(raw_path)
        mask = sitk.ReadImage(mask_path)
        raw_img = sitk.GetArrayFromImage(raw)
        mask_img = sitk.GetArrayFromImage(mask)
        masked = np.where(mask_img==0, bg_value, raw_img)

        masked = sitk.GetImageFromArray(masked)
        masked.CopyInformation(raw)
        sitk.WriteImage(masked, os.path.join(out_dir, os.path.basename(raw_path)))


def create_body_mask_dir(raw_dir, out_dir, filetype):
    """create body mask for all images in raw_dir"""
    for raw_path in tqdm(glob.glob(os.path.join(raw_dir, f"*.{filetype}"))):
        dst = os.path.join(out_dir, os.path.basename(raw_path))
        create_body_mask(raw_path, dst)

def create_body_mask(in_img, out_mask):
    """
    Create a body mask
    Adapted from https://github.com/MASILab/thorax_level_BCA/blob/0ff54db11395a28b62d91132be0df49e4927a3b5/Utils/utils.py#L577
    handles sitk images
    """
    rBody = 2

    print(f'Get body mask of image {in_img}')
    image_sitk = sitk.ReadImage(in_img)
    image_np = sitk.GetArrayFromImage(image_sitk)
    image_np = ndi.gaussian_filter(image_np, sigma=1.5) # blur improves algo for noisy images
    BODY = (image_np >= -500)  # & (I<=win_max)
    print(f'{np.sum(BODY)} of {np.size(BODY)} voxels masked.')
    if np.sum(BODY) == 0:
        raise ValueError('BODY could not be extracted!')

    # Find largest connected component in 3D
    struct = np.ones((3, 3, 3), dtype=np.bool)
    BODY = ndi.binary_erosion(BODY, structure=struct, iterations=rBody)

    BODY_labels = skimage.measure.label(np.asarray(BODY, dtype=int))

    props = skimage.measure.regionprops(BODY_labels)
    areas = []
    for prop in props:
        areas.append(prop.area)
    print(f' -> {len(areas)} areas found.')
    # only keep largest, dilate again and fill holes
    BODY = ndi.binary_dilation(BODY_labels == (np.argmax(areas) + 1), structure=struct, iterations=rBody)
    # Fill holes slice-wise
    for z in range(0, BODY.shape[2]):
        BODY[:, :, z] = ndi.binary_fill_holes(BODY[:, :, z])
    for y in range(0, BODY.shape[1]):
        BODY[:, y, :] = ndi.binary_fill_holes(BODY[:, y, :])
    for x in range(0, BODY.shape[0]):
        BODY[x, :, :] = ndi.binary_fill_holes(BODY[x, :, :])

    new_image = sitk.GetImageFromArray(BODY.astype(np.int8))
    new_image.CopyInformation(image_sitk)
    sitk.WriteImage(new_image, out_mask)
    # new_image = nib.Nifti1Image(BODY.astype(np.int8), header=image_nb.header, affine=image_nb.affine)
    # nib.save(new_image, out_mask)
    print(f'Generated body_mask segs in Abwall {out_mask}')
            
if __name__ == "__main__":
    # copy_dataset(*sys.argv[1:])
    # train_test_split(*sys.argv[1:])
    # fix_labels(*sys.argv[1:])
    # create_body_mask_dir(*sys.argv[1:])
    # apply_body_mask_dir(*sys.argv[1:])
    generate_kfolds(*sys.argv[1:])
    # get_kfolds(*sys.argv[1:])
    # generate_clips(*sys.argv[1:])