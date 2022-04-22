"""Dataloaders for Lobe segmentation with MONAI"""

from monai.utils import set_determinism
from monai.transforms import (
    EnsureChannelFirstd,
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
    EnsureTyped
)
from monai.data import DataLoader, Dataset
import numpy as np
import os

# unwrap directory paths
def train_dataloader(config, train_images):
    LABEL_DIR = config["label_dir"]

    # get labels from vlsp
    if config["dataset"] == "vlsp":
        train_file_names = [f"lvlsetseg_{os.path.basename(name)}" for name in train_images]
    elif config["dataset"] == "luna16":
        train_file_names = [f"{os.path.basename(name)[:-4]}_LobeSegmentation.nrrd" for name in train_images]
    else:
        print("Error: define dataset in Config.YAML")
        return

    train_labels = [os.path.join(LABEL_DIR, name) for name in train_file_names]

    train_files = [
        {"image": image_name, "label": label_name}
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
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=hu_window[0], a_max=hu_window[1], b_min=0.0, b_max=1.0,
                             clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=CROP_SHAPE,
            pos=1, # prob of picking a voxel
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

def val_dataloader(config, val_images):
    LABEL_DIR = config["label_dir"]

    if config["dataset"] == "vlsp":
        val_file_names = [f"lvlsetseg_{os.path.basename(name)}" for name in val_images]
    elif config["dataset"] == "luna16":
        val_file_names = [f"{os.path.basename(name)[:-4]}_LobeSegmentation.nrrd" for name in val_images]
    else:
        print("Error: define dataset in Config.YAML")
        return
    val_labels = [os.path.join(LABEL_DIR, name) for name in val_file_names]

    val_files = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(val_images, val_labels)
    ]

    set_determinism(seed=config["random_seed"])

    # Hyperparams and constants
    BATCH_SIZE = config["batch_size"]
    CROP_SHAPE = config["crop_shape"]  # produce 4 crops of this size from raw image
    # Transforms
    hu_window = config["window"]  # lung Hounsfield Unit window
    # Transforms
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        # EnsureChannelFirstd(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=config["pix_dim"], mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=hu_window[0], a_max=hu_window[1], b_min=0.0, b_max=1.0,
                                 clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)
    print(f"Validation sample size: {len(val_ds)}")

    return val_loader
