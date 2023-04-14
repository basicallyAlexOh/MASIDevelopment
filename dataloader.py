"""Dataloaders for Lobe segmentation with MONAI"""

from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    AddChanneld,
    RandShiftIntensityd,
    RandAffined,
    ToTensord,
    EnsureTyped,
    SpatialPadd,
    SpatialPad,
)
from monai.data import DataLoader, Dataset
import numpy as np
import os
from skimage.transform import resize
import math
import torch

def npy_train_loader(config, npys):
    label_dir = config["label_dir"]
    labels = [os.path.join(label_dir, os.path.basename(npy)) for npy in npys]
    files = [
        {"image": npy, "label": label}
        for npy, label in zip(npys, labels)
    ]
    npy_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        SpatialPadd(keys=["image", "label"], spatial_size=config["crop_shape"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=config["crop_shape"],
            pos=0.8, # prob of picking a positive voxel
            neg=0,
            num_samples=config["crop_nsamples"],
            image_key="image",
            image_threshold=0,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.20,
        ),
        RandAffined(
            keys=['image', 'label'],
            mode=('bilinear', 'nearest'),
            prob=1.0, spatial_size=config["crop_shape"],
            rotate_range=(0, 0, np.pi / 30),
            scale_range=(0.1, 0.1, 0.1)),
        ToTensord(keys=["image", "label"]),
    ])

    set_determinism(seed=config["random_seed"])
    train_ds = Dataset(data=files, transform=npy_transforms)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                        num_workers=config["num_workers"], pin_memory=True)
    return train_loader

def npy_test_loader(config, npys):
    label_dir = config["label_dir"]
    labels = [os.path.join(label_dir, os.path.basename(npy)) for npy in npys]
    files = [
        {"image": npy, "label": label, "image_path": npy}
        for npy, label in zip(npys, labels)
    ]
    test_transforms = Compose([
        LoadImaged(keys=["image"]),
        ToTensord(keys=["image"]),
    ])

    set_determinism(seed=config["random_seed"])
    test_ds = Dataset(data=files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True,
                        num_workers=config["num_workers"], pin_memory=True)
    return test_loader

# unwrap directory paths
def train_dataloader(config, train_images):
    LABEL_DIR = config["label_dir"]

    # get labels from vlsp
    if config["dataset"] == "vlsp":
        train_file_names = [f"lvlsetseg_{os.path.basename(name)}" for name in train_images]
    elif config["dataset"] == "TS":
        train_file_names = [os.path.basename(name) for name in train_images]
    elif config["dataset"] == "luna16":
        train_file_names = [f"{os.path.basename(name)[:-4]}_LobeSegmentation.nrrd" for name in train_images]
    elif config["dataset"] == "mixed":
        train_file_names = []
        for i in train_images:
            name, suffix = os.path.splitext(os.path.basename(i))
            if suffix == ".mhd":
                train_file_names.append(f"{name}_LobeSegmentation.nrrd")
            elif suffix == ".gz":
                fname = f"{name[:-4]}_LobeSegmentation.nii.gz" if name[1] == '.' else f"{name[:-4]}_lvlsetseg.nii.gz"
                train_file_names.append(fname)
    else:
        print("Error: define dataset in Config.YAML")
        return
    train_labels = [os.path.join(LABEL_DIR, name) for name in train_file_names]
    train_files = [
        {"image": image_name, "label": label_name, "image_path": image_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    # val_size = int(len(train_images)*config["val_ratio"])
    # train_files, val_files = data_dicts[:-val_size], data_dicts[-val_size:]

    set_determinism(seed=config["random_seed"])

    # Hyperparams and constants
    BATCH_SIZE = config["batch_size"]
    CROP_SHAPE = config["crop_shape"] # produce 4 crops of this size from raw image

    # Transforms
    hu_window = config["window"] # lung Hounsfield Unit window

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        # EnsureChannelFirstd(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=config["pix_dim"], mode=("bilinear", "nearest")),
        MatchSized(keys=["image", "label"], mode="crop"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=hu_window[0], a_max=hu_window[1], b_min=0.0, b_max=1.0,
                            clip=True),
        # CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=CROP_SHAPE),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=CROP_SHAPE,
            pos=0.8, # prob of picking a positive voxel
            neg=0,
            num_samples=config["crop_nsamples"],
            image_key="image",
            image_threshold=0,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.20,
        ),
        RandAffined(
            keys=['image', 'label'],
            mode=('bilinear', 'nearest'),
            prob=1.0, spatial_size=CROP_SHAPE,
            rotate_range=(0, 0, np.pi / 30),
            scale_range=(0.1, 0.1, 0.1)),
        ToTensord(keys=["image", "label"]),
    ])

    # Initialize Dataset
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=config["num_workers"], pin_memory=True)

    print(f"Training sample size: {len(train_ds)}")

    return train_loader

def get_val_transforms(config):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=config["pix_dim"], mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=config["window"][0], a_max=config["window"][1], 
            b_min=0.0, b_max=1.0, clip=True),
        # CropForegroundd(keys=["image", "label"], source_key="image"),
        MatchSized(keys=["image", "label"], mode="interp"),
        ToTensord(keys=["image", "label"]),
    ])

def val_dataloader(config, val_images):
    LABEL_DIR = config["label_dir"]

    if config["dataset"] == "vlsp":
        val_file_names = [f"lvlsetseg_{os.path.basename(name)}" for name in val_images]
    elif config["dataset"] == "TS":
        val_file_names = [os.path.basename(name) for name in val_images]
    elif config["dataset"] == "luna16":
        val_file_names = [f"{os.path.basename(name)[:-4]}_LobeSegmentation.nrrd" for name in val_images]
    elif config["dataset"] == "mixed":
        val_file_names = []
        for i in val_images:
            name, suffix = os.path.splitext(os.path.basename(i))
            if suffix == ".mhd":
                val_file_names.append(f"{name}_LobeSegmentation.nrrd")
            elif suffix == ".gz":
                fname = f"{name[:-4]}_LobeSegmentation.nii.gz" if name[1] == '.' else f"{name[:-4]}_lvlsetseg.nii.gz"
                val_file_names.append(fname)
    else:
        print("Error: define dataset in Config.YAML")
        return
    val_labels = [os.path.join(LABEL_DIR, name) for name in val_file_names]

    val_files = [
        {"image": image_name, "label": label_name, "image_path": image_name}
        for image_name, label_name in zip(val_images, val_labels)
    ]

    # Transforms
    val_transforms = get_val_transforms(config)

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, shuffle=False)
    print(f"Validation sample size: {len(val_ds)}")

    return val_loader

def test_dataloader(config, val_images):
    # test_transforms = Compose([
    #     LoadImaged(keys=["image", "label"]),
    #     AddChanneld(keys=["image", "label"]),
    #     Spacingd(keys=["image", "label"], pixdim=config["pix_dim"], mode=("bilinear", "nearest")),
    #     Orientationd(keys=["image", "label"], axcodes="RAS"),
    #     ScaleIntensityRanged(keys=["image"], a_min=config["window"][0], a_max=config["window"][1], b_min=0.0, b_max=1.0,
    #                          clip=True),
    #     # CropForegroundd(keys=["image", "label"], source_key="image"),
    #     EnsureTyped(keys=["image", "label"]),
    # ])
    test_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image"], pixdim=config["pix_dim"], mode=("bilinear")),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=config["window"][0], a_max=config["window"][1], b_min=0.0, b_max=1.0,
                            clip=True),
        EnsureTyped(keys=["image"]),
    ])
    LABEL_DIR = config["test_label_dir"]

    if config["dataset"] == "vlsp":
        val_file_names = [f"lvlsetseg_{os.path.basename(name)}" for name in val_images]
    elif config["dataset"] == "TS":
        val_file_names = [os.path.basename(name) for name in val_images]
    elif config["dataset"] == "luna16":
        val_file_names = [f"{os.path.basename(name)[:-4]}_LobeSegmentation.nrrd" for name in val_images]
    elif config["dataset"] == "mixed":
        val_file_names = []
        for i in val_images:
            name, suffix = os.path.splitext(os.path.basename(i))
            if suffix == ".mhd":
                val_file_names.append(f"{name}_LobeSegmentation.nrrd")
            elif suffix == ".vscgz":
                fname = f"{name[:-4]}_LobeSegmentation.nii.gz" if name[1] == '.' else f"{name[:-4]}_lvlsetseg.nii.gz"
                val_file_names.append(fname)
    else:
        print("Error: define dataset in Config.YAML")
        return
    val_labels = [os.path.join(LABEL_DIR, name) for name in val_file_names]
    val_files = [
        {"image": image_name, "label": label_name, "image_path": image_name}
        for image_name, label_name in zip(val_images, val_labels)
    ]
    test_ds = Dataset(data=val_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=2, shuffle=False)
    return test_loader

def infer_dataloader(config, val_images):
    test_transforms = Compose([
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        Spacingd(keys=["image"], pixdim=config["pix_dim"], mode=("bilinear")),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=config["window"][0], a_max=config["window"][1], b_min=0.0, b_max=1.0,
                            clip=True),
        EnsureTyped(keys=["image"]),
    ])
    val_files = [
        {"image": image_name, "image_path": image_name}
        for image_name in val_images
    ]
    infer_ds = Dataset(data=val_files, transform=test_transforms)
    infer_loader = DataLoader(infer_ds, batch_size=1, num_workers=config["num_workers"], shuffle=False)
    return infer_loader

class MatchSized(object):
    """Resize input A to match size of input B"""
    def __init__(self, keys, mode="interp"):
        self.keyA, self.keyB = keys[0], keys[1]
        self.mode = mode
    
    def __call__(self, data):
        a, b = data[self.keyA], data[self.keyB]
        if a.shape != b.shape:
            if self.mode=="interp":
                a = resize(a, b.shape)
            else:
                # crop then pad
                a = a[..., :b.shape[-3], :b.shape[-2], :b.shape[-1]]
                pad = SpatialPad(b.shape[-3:])
                a = pad(a)
        # assert (math.prod(a.shape) > 96**3), "less than (96,96,96) patch shape"
        assert (a.shape==b.shape), f"resizing failed: data are not same shape! {data['image_path']}: {a.shape}, {b.shape}"
        return {self.keyA: a, self.keyB: b}

# if __name__ == "__main__":
#     import nibabel as nib
#     img = nib.load("/home-nfs2/local/VANDERBILT/litz/data/imagevu/nifti/train/00000001time20131205.nii.gz").get_fdata()
#     label = nib.load("/home-nfs2/local/VANDERBILT/litz/data/imagevu/lobe/lvlsetsegCC/lvlsetseg_00000001time20131205.nii.gz").get_fdata()
#     print(img.shape, label.shape)
#     data = {"img": "/home-nfs2/local/VANDERBILT/litz/data/imagevu/nifti/train/00000001time20131205.nii.gz", "label": "/home-nfs2/local/VANDERBILT/litz/data/imagevu/lobe/lvlsetsegCC/lvlsetseg_00000001time20131205.nii.gz"}
#     resize_tf = Compose([
#         LoadImaged(keys=["img", "label"]),
#         AddChanneld(keys=["img", "label"]),
#         Spacingd(keys=["img", "label"], pixdim=(1,1,1), mode=("bilinear", "nearest")),
#         MatchSized(keys=["img", "label"], mode="crop"),
#         ])
#     data_tf = resize_tf(data)
#     print(data_tf["img"].shape)
