import os
from torch.utils.data import Dataset, DataLoader
import nibabel as nib

class NiftiDataset(Dataset):
    def __init__(self, dataset_dir, label_dir, sample_size=None, transforms=None):
        self.dataset_dir = dataset_dir
        self.label_dir = label_dir
        self.transforms=transforms
        self.data_list = os.listdir(self.dataset_dir)
        self.label_list = os.listdir(self.label_dir)
        if sample_size:
            self.data_list = self.data_list[:sample_size]
            self.label_dir = self.label_list[:sample_size]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        nii_path = os.path.join(self.dataset_dir, self.data_list[idx])
        label_path = os.path.join(self.label_dir, self.label_list[idx])
        img = nib.load(nii_path).get_fdata()
        label = nib.load(label_path).get_fdata()
        # apply transform to both img and label mask
        if self.transform:
            for transform in self.transforms:
                img = transform(img)
                label = transform(label)

        return img, label


class Resize(img, shape):
    """ Resize volume to a given shape"""
    