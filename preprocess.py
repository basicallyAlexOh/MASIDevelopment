import os
import sys
import random
import shutil
import nibabel as nib
import numpy as np
from tqdm import tqdm

def copy_raw(root_dir, train_dir, test_dir):
    """Split imagevu into training and test set (8:2 ratio by patient) and copy over to local"""
    pids = os.listdir(root_dir)
    n_pids = len(pids)
    random.seed(10)
    train = random.sample(pids, round(n_pids*0.8))
    test = [x for x in pids if x not in train]

    for pid in tqdm(pids):
        for sessionid in os.listdir(os.path.join(root_dir, pid)):
            for nii_fname in os.listdir(os.path.join(root_dir, pid, sessionid)):
                nii_path = os.path.join(root_dir,pid,sessionid,nii_fname)
                if pid in train:
                    shutil.copyfile(nii_path, os.path.join(train_dir, nii_fname))
                if pid in test:
                    shutil.copyfile(nii_path, os.path.join(test_dir, nii_fname))

def apply_body_mask(mask_dir, raw_dir, out_dir, bg_value=-1500):
    """ Uses existing body mask to set background pixels to some HU value"""
    for fname in tqdm(os.listdir(raw_dir)):
        raw_path = os.path.join(raw_dir, fname)
        mask_path = os.path.join(mask_dir, fname)
        raw = nib.load(raw_path)
        mask = nib.load(mask_path)
        raw_img = raw.get_fdata()
        mask_img = mask.get_fdata()
        masked = np.where(mask_img==0, bg_value, raw_img)

        masked = nib.Nifti1Image(masked, header=raw.header, affine=raw.affine)
        nib.save(masked, os.path.join(out_dir, fname))


if __name__ == '__main__':
    ROOT_DIR = "/nfs/masi/SPORE/nifti/combine/"
    TRAIN_DIR = "/home/local/VANDERBILT/litz/data/imagevu/nifti/train/"
    TEST_DIR = "/home/local/VANDERBILT/litz/data/imagevu/nifti/test/"
    MASK_DIR = "/nfs/masi/xuk9/Projects/ThoraxLevelBCA/VLSP_all/body_mask"
    #copy_raw(ROOT_DIR, TRAIN_DIR, TEST_DIR)
    apply_body_mask(MASK_DIR, sys.argv[1], sys.argv[2], -1500)