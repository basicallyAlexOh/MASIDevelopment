"""Test baseline methods (JoHof model, level-set) as comparison"""

import argparse
import sys
import os
import glob
import SimpleITK as sitk
from pandas import DataFrame
import numpy as np
from monai.utils import set_determinism
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    Spacingd,
    EnsureTyped,
    AsDiscrete,
    EnsureType,
)
from monai.data import DataLoader, Dataset, decollate_batch
from monai.metrics import DiceMetric
import torch
from tqdm import tqdm
from luna16_preprocess import get_kfolds

def test_johof_luna16(test_dir, seg_dir, label_dir, metrics_f):
    """Compute test metrics on test set"""
    # only test on the 15 test cases
    test_cases = sorted(glob.glob(os.path.join(test_dir, "*.mhd")))

    # retrieve seg and labels that match test cases
    # seg_names = [f"johof_fused_{os.path.basename(name)}" for name in test_cases]
    seg_names = [f"lvlsetseg_{os.path.basename(name)}" for name in test_cases]
    segs = [os.path.join(seg_dir, name) for name in seg_names]
    label_names = [f"{os.path.basename(name)[:-4]}_LobeSegmentation.nrrd" for name in test_cases]
    labels = [os.path.join(label_dir, name) for name in label_names]

    test_loader = test_dataloader(segs, labels)
    test_metric = DiceMetric(include_background=False, reduction="none")

    device = torch.device("cuda:0")
    post_pred = Compose([EnsureType(), AsDiscrete(to_onehot=6)])
    post_labels = Compose([AsDiscrete(to_onehot=6)])

    for test_data in tqdm(test_loader):
        seg, label = (test_data["image"].to(device), test_data["label"].to(device))
        # print(decollate_batch(seg))
        seg = [post_pred(i) for i in decollate_batch(seg)]
        # print(seg[0].shape)
        label = [post_labels(i) for i in decollate_batch(label)]
        # print(label[0].shape)
        test_metric(y_pred=seg, y=label)

    test_dices = test_metric.aggregate()

    # Record metrics and compute mean over test set
    class_means = torch.mean(test_dices, dim=0)
    mean = torch.mean(test_dices)
    test_dices_df = DataFrame(test_dices.detach().cpu().numpy())
    test_dices_df.to_csv(metrics_f)

    print(f"Average class scores: {class_means}")
    print(f"Average score overall: {mean}")

def cv_test_johof_luna16(k, kfolds_path, seg_dir, label_dir, metrics_f, model=0):
    """
    Compute test metrics for k folds
    :param
    model: 0 = johof, 1 = LSM
    """
    k = int(k)
    images = get_kfolds(kfolds_path)
    test_images = images[k-1]
    labels = [f"{os.path.basename(name)[:-4]}_LobeSegmentation.nrrd" for name in test_images]
    labels = [os.path.join(label_dir, name) for name in labels]

    if model==0:
        seg_names = [f"johof_fused_{os.path.basename(name)}" for name in test_images]
        segs = [os.path.join(seg_dir, name) for name in seg_names]
        test_loader = test_dataloader(segs, labels, normalize_spacing=False)
    else:
        seg_names = [f"lvlsetseg_{os.path.basename(name)}" for name in test_images]
        segs = [os.path.join(seg_dir, name) for name in seg_names]
        test_loader = test_dataloader(segs, labels, normalize_spacing=True)

    test_metric = DiceMetric(include_background=False, reduction="none")
    device = torch.device("cuda:0")
    post_pred = Compose([EnsureType(), AsDiscrete(to_onehot=6)])
    post_labels = Compose([AsDiscrete(to_onehot=6)])
    for test_data in tqdm(test_loader):
        seg, label = (test_data["image"].to(device), test_data["label"].to(device))
        # print(seg.shape)
        # print(label.shape)
        seg = [post_pred(i) for i in decollate_batch(seg)]
        label = [post_labels(i) for i in decollate_batch(label)]
        test_metric(y_pred=seg, y=label)

    test_dices = test_metric.aggregate()

    # Record metrics and compute mean over test set
    class_means = torch.mean(test_dices, dim=0)
    mean = torch.mean(test_dices)
    test_dices_df = DataFrame(test_dices.detach().cpu().numpy())
    test_dices_df.to_csv(metrics_f)
    print(f"Average class scores: {class_means}")
    print(f"Average score overall: {mean}")

def test_dataloader(segs, labels, normalize_spacing=False):
    test_files = [
        {"image": seg_name, "label": label_name}
        for seg_name, label_name in zip(segs, labels)
    ]

    set_determinism(1)

    # if normalize_spacing:
    test_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1,1,1), mode=("bilinear", "nearest")),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])
    # else:
    #     test_transforms = Compose([
    #         LoadImaged(keys=["image", "label"]),
    #         EnsureChannelFirstd(keys=["image", "label"]),
    #         Orientationd(keys=["image", "label"], axcodes="RAS"),
    #         EnsureTyped(keys=["image", "label"]),
    #     ])
    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)
    print(f"Test sample size: {len(test_ds)}")
    return test_loader

def dice(im1, im2):
    """
    https://gist.github.com/JDWarner/6730747
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--kfolds-path', type=str, default='/home/local/VANDERBILT/litz/data/luna16/5fold_qa.csv')
    parser.add_argument('--seg-dir', type=str, default='/home/local/VANDERBILT/litz/data/luna16/johof_fused')
    parser.add_argument('--label-dir', type=str, default='/home/local/VANDERBILT/litz/data/luna16/fixed_labels')
    parser.add_argument('--metrics-f', type=str, default='/home/local/VANDERBILT/litz/github/MASILab/lobe_seg/tmp/test.csv')
    parser.add_argument('--model', type=int, default=0) # 0 = johof, 1 = LSM
    args = parser.parse_args()
    print(args)
    # test_johof_luna16(*sys.argv[1:])
    cv_test_johof_luna16(args.k, args.kfolds_path, args.seg_dir, args.label_dir, args.metrics_f, args.model)