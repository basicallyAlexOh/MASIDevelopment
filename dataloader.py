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
    AsDiscreted,
    Invertd,
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
            pos=1, # prob of picking a positive voxel
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
        # EnsureChannelFirstd(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=config["pix_dim"], mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=config["window"][0], a_max=config["window"][1], b_min=0.0, b_max=1.0,
                                 clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])

def val_dataloader(config, val_images):
    LABEL_DIR = config["label_dir"]

    if config["dataset"] == "vlsp":
        val_file_names = [f"lvlsetseg_{os.path.basename(name)}" for name in val_images]
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
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(val_images, val_labels)
    ]

    # Transforms
    val_transforms = get_val_transforms(config)

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, shuffle=False)
    print(f"Validation sample size: {len(val_ds)}")

    return val_loader

def test_dataloader(config, val_images):
    test_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=config["pix_dim"], mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=config["window"][0], a_max=config["window"][1], b_min=0.0, b_max=1.0,
                             clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])
    invert_transforms = Compose([
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=test_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        # AsDiscreted(keys="pred", argmax=True, to_onehot=6),
        # AsDiscreted(keys="label", to_onehot=6)
    ])
    LABEL_DIR = config["label_dir"]

    if config["dataset"] == "vlsp":
        val_file_names = [f"lvlsetseg_{os.path.basename(name)}" for name in val_images]
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
    test_ds = Dataset(data=val_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=2, shuffle=False)
    return test_loader, invert_transforms