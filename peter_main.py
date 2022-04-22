#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 11:06:19 2021

@author: leeh43
"""

from monai.utils import set_determinism, first
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    SpatialPadd,
    ResizeWithPadOrCropd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    KeepLargestConnectedComponentd,
    Spacingd,
    ToTensord,
    ToTensor,
    RandAffined,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandRotate90d
)

# from monai.networks.nets import UNet
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, compute_meandice
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import numpy as np
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"]='0'

out_classes = 6
img_dir = os.path.join('/home/local/VANDERBILT/litz/data/imagevu/nifti/train_preproc/')
label_dir = os.path.join('/home/local/VANDERBILT/litz/data/imagevu/lobe/lvlsetsegCC/')

all_images = []
all_labels = []
for img in os.listdir(img_dir):
    img_file = os.path.join(img_dir, img)
    label_file = os.path.join(label_dir, 'lvlsetseg_' + img)

    all_images.append(img_file)
    all_labels.append(label_file)

# v_img_path = os.path.join('/nfs/masi/leeh43/BTCV_NC/imagesTs')
# v_label_path = os.path.join('/nfs/masi/leeh43/BTCV_NC/labelsTs')

# all_images = sorted(glob.glob(os.path.join(t_img_path, "*.nii.gz")))
# all_labels = sorted(glob.glob(os.path.join(t_label_path, "*.nii.gz")))

# valid_images = sorted(glob.glob(os.path.join(v_img_path, "*.nii.gz")))
# valid_labels = sorted(glob.glob(os.path.join(v_label_path, "*.nii.gz")))

# print(sorted(valid_images))
# all_images = sorted(train_images) + sorted(valid_images)
# print(len(train_images), len(valid_images))
# all_labels = sorted(train_labels) + sorted(valid_labels)
# print(len(train_labels), len(valid_labels))

data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(all_images, all_labels)
]
# train_files, val_files = data_dicts[:80], data_dicts[80:]
train_files, val_files = data_dicts[:int(len(all_images)*0.8)], data_dicts[int(len(all_images)*0.8):]
set_determinism(seed=0)

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(
            1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        # ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(168,168,128), mode=("constant")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1024, a_max=600,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        # user can also add other random transforms
        # RandFlipd(
        #     keys=["image", "label"],
        #     spatial_axis=[0],
        #     prob=0.10,
        # ),
        # RandFlipd(
        #     keys=["image", "label"],
        #     spatial_axis=[1],
        #     prob=0.10,
        # ),
        # RandFlipd(
        #     keys=["image", "label"],
        #     spatial_axis=[2],
        #     prob=0.10,
        # ),
        # RandRotate90d(
        #     keys=["image", "label"],
        #     prob=0.10,
        #     max_k=3,
        # ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.20,
        ),
        RandAffined(
            keys=['image', 'label'],
            mode=('bilinear', 'nearest'),
            prob=1.0, spatial_size=(96, 96, 96),
            rotate_range=(0, 0, np.pi/30),
            scale_range=(0.1, 0.1, 0.1)),
        ToTensord(keys=["image", "label"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(
            1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        # ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(168,168,128), mode=("constant")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1024, a_max=600,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)

## Check Dataset
# check_ds = Dataset(data=train_files, transform=train_transforms)
# check_loader = DataLoader(check_ds, batch_size=4)
# check_data = first(check_loader)
# image, label = (check_data["image"][2][0], check_data["label"][2][0])
# print(f"image shape: {image.shape}, label shape: {label.shape}")
# plot the slice [:, :, 80]
# fig = plt.figure("check", (12, 6))
# plt.subplot(1, 2, 1)
# plt.title("image")
# plt.imshow(image[:, :, 110], cmap="gray")
# plt.subplot(1, 2, 2)
# plt.title("label")
# ax = plt.imshow(label[:, :, 110])
# fig.colorbar(ax)
# plt.show()

train_ds = Dataset(data=train_files, transform=train_transforms)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)

val_ds = Dataset(data=val_files, transform=val_transforms)
# val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)

device = torch.device("cuda:0")
model = UNet(spatial_dims=3,
        in_channels=1,
        out_channels=out_classes,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
).to(device)

# Resume training from checkpoint if indicated
CHECKPOINT = False
PRETRAINED = False

# Finetune pretrained model if indicated
if PRETRAINED:
    print(f"Fine tuning model from {PRETRAINED}")
    model.load_state_dict(torch.load(PRETRAINED))

# loss_function = DiceLoss(to_onehot_y=True, softmax=True)
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
# dice_metric = DiceMetric(include_background=False, reduction='mean')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1000)
# max_epochs = 600
# val_interval = 1
# best_metric = -1
# best_metric_epoch = -1
# epoch_loss_values = []
# metric_values = []
# post_pred = Compose([ToTensor(), AsDiscrete(argmax=True, to_onehot=True, n_classes=num_classes )])
# post_label = Compose([ToTensor(), AsDiscrete(to_onehot=True, n_classes=num_classes )])

# root_dir = os.path.join('/nfs/masi/leeh43/thomas_dataset/train_model/all_bz_2_DiceCE')
root_dir = os.path.join('/home/local/VANDERBILT/litz/github/MASILab/lobe_seg/peter/unet512_0416')
if os.path.exists(root_dir) == False:
    os.makedirs(root_dir)
    
t_dir = os.path.join(root_dir, 'tensorboard')
if os.path.exists(t_dir) == False:
    os.makedirs(t_dir)
writer = SummaryWriter(log_dir=t_dir)


def validation(epoch_iterator_val):
    model.eval()
    dice_vals = list()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
            )
        dice_metric.reset()
    mean_dice_val = np.mean(dice_vals)
    writer.add_scalar('Validation Segmentation Loss', mean_dice_val, global_step)
    
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(
                    model.state_dict(), os.path.join(root_dir, "best_metric_model.pth")
                )
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
                # scheduler.step(dice_val)
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
                # scheduler.step(dice_val)
        writer.add_scalar('Training Segmentation Loss', loss.data, global_step)
        global_step += 1

    return global_step, dice_val_best, global_step_best


max_iterations = 40000
eval_num = 500
post_label = AsDiscrete(to_onehot=True, n_classes=out_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=out_classes)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best
    )





