
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