"""Evaluate model on test set and report results"""

import os
import sys
from tqdm import tqdm
import torch
import SimpleITK as sitk
import gc
import psutil
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import (
    Compose,
    AsDiscrete,
    AddChannel,
    EnsureType,
    EnsureTyped,
    AsDiscreted,
    Spacingd,
    Resized,
    Orientationd,
)
from pandas import DataFrame
import numpy as np
import nibabel as nib
from postprocess import get_largest_cc, lungmask_filling, nearest_label_filling
sys.path.append("/home/local/VANDERBILT/litz/github/MASILab/thoraxtools/func")
import vis.paral_clip_overlay_mask as overlay

def test(config,
         config_id,
         device,
         model,
         model_path,
         test_metric,
         test_loader,
         metrics_path,
         seg_dir,
         clip_dir):

    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_metric.reset()

    measure_transforms = Compose([EnsureType(), AddChannel(), AsDiscrete(to_onehot=6)])
    image_paths = []
    # reader = sitk.ImageFileReader()
    # print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    with torch.no_grad():
        for test_data in tqdm(test_loader):
            test_inputs, image_path = (
                test_data["image"].to(device),
                test_data["image_path"][0],
            )
            # make transforms from image metadata
            raw_nii = nib.load(image_path)
            axcodes = nib.orientations.aff2axcodes(raw_nii.affine)
            axcodes = ''.join(axcodes)
            pixdim = raw_nii.header.get_zooms()
            spatial_size = raw_nii.shape
            post_pred_transforms = Compose([
                EnsureTyped(keys=["label", "pred"]),
                AsDiscreted(keys="pred", argmax=True, to_onehot=6),
                AsDiscreted(keys="label", to_onehot=6),
                Orientationd(keys=["pred"], axcodes=axcodes),
                Spacingd(keys=["pred"], pixdim=pixdim, mode="nearest"), 
                Resized(keys=["pred"], spatial_size=spatial_size, mode="nearest"),
            ])

            image_paths.append(image_path)
            test_data["pred"] = sliding_window_inference(test_inputs, config["crop_shape"], 4, model)
            test_data["pred"] = test_data["pred"].detach().cpu() # image space transform is mem intensive
            test_data = [post_pred_transforms(i) for i in decollate_batch(test_data)]
            test_data = test_data[0]

            # postprocces with CC and nearest label filling
            pred = torch.argmax(test_data["pred"], dim=0)
            pred = lungmask_filling(get_largest_cc(pred), image_path)
            test_data["pred"] = measure_transforms(pred)
            
            # compute dice
            test_metric(y_pred=[test_data["pred"]], y=[test_data["label"]])
            # final label map
            label_map = torch.argmax(test_data["pred"], dim=0)

            # dice of this example's RML
            # rml_dice = test_metric.aggregate()[-1][3]

            # pred_img_space = invert_transforms(test_data)["pred"]
            # pred_img_space = torch.argmax(pred_img_space, dim=0)
            #
            # if seg_dir and rml_dice < 0.7:
            # # # if seg_dir:
            #     print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
            #     reader.SetFileName(image_path)
            #     reader.ReadImageInformation()
            #     seg = np.transpose(pred_img_space, (2, 1, 0))
            #     seg = sitk.GetImageFromArray(seg)
            #     seg.SetOrigin(reader.GetOrigin())
            #     seg.SetSpacing(reader.GetSpacing())
            #     seg.SetDirection(reader.GetDirection())
            #     sitk.WriteImage(seg, os.path.join(seg_dir, os.path.basename(image_path)))
            #     print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
            # if clip_dir and rml_dice < 0.7:
            # # if clip_dir:
            #     # raw_sitk = sitk.ReadImage(image_path)
            #     # raw = sitk.GetArrayFromImage(raw_sitk)
            #     # raw = test_data["image"]
            #     raw = test_data["image"].detach().cpu().numpy()
            #     raw = raw[0,:,:,:]
            #     # fname = os.path.basename(image_path)[:-4] if config["dataset"]=="luna16" else os.path.basename(image_path).split(".")[0]
            #     fname = os.path.basename(image_path)[:-7]
            #     overlay.multiple_clip_overlay_with_mask_from_npy(raw, postprocess, os.path.join(clip_dir, f"pred_{fname}_coronal.png"), clip_plane='coronal', img_vrange=(0, 1))
            #     overlay.multiple_clip_overlay_with_mask_from_npy(raw, label, os.path.join(clip_dir, f"label_{fname}_coronal.png"), clip_plane='coronal', img_vrange=(0, 1))
            #     overlay.multiple_clip_overlay_with_mask_from_npy(raw, postprocess,
            #                                                      os.path.join(clip_dir, f"pred_{fname}_axial.png"),
            #                                                      clip_plane='axial', img_vrange=(0, 1))
            #     overlay.multiple_clip_overlay_with_mask_from_npy(raw, label,
            #                                                  os.path.join(clip_dir, f"label_{fname}_axial.png"),
            #                                                  clip_plane='axial', img_vrange=(0, 1))
                # overlay.multiple_clip_overlay_with_mask_from_npy(raw, postprocess, os.path.join(clip_dir, f"pred_{fname}_sagittal.png"), clip_plane='sagittal', img_vrange=(0,1))
                # overlay.multiple_clip_overlay_with_mask_from_npy(raw, postprocess, os.path.join(clip_dir, f"pred_{fname}_axial.png"), clip_plane='axial', img_vrange=(0,1))
                # print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
            gc.collect()

        # Total dice over test set
        test_dices = test_metric.aggregate()

        # Record metrics and compute mean over test set
        class_means = torch.mean(test_dices, dim=0)
        mean = torch.mean(test_dices)

        # store in dataframe with image path
        test_dices_df = DataFrame(test_dices.detach().cpu().numpy())
        test_dices_df["input_path"] = image_paths
        test_dices_df.to_csv(metrics_path)

    # Log best dice
    # print(f"All scores: {test_dices_df}")
    print(f"Average class scores: {class_means}")
    print(f"Average score overall: {mean}")

