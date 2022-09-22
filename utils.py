
import os
import re
import numpy as np
import SimpleITK as sitk

def parse_id(name, suffix=".nii.gz"):
    scanid = name.split("_")[0]
    return scanid

def sitk2np(path):
    img = sitk.GetArrayFromImage(sitk.ReadImage(path))
    img = np.transpose(img, (2, 1, 0))
    return img

def read_image(image_path):
    """
    Return SITK object if NOT .npy format
    Return np array if .npy format
    Assumes no . in image name
    """
    suffix = image_path.split(".")[1]
    if suffix=="npy":
        return np.load(image_path)
    else:
        return sitk.ReadImage(image_path)
