import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import NiftiDataset
from model import Unet3d
from losses import cross_entropy_loss, dice_coeff

TRAIN_DIR = "/home/local/VANDERBILT/litz/data/imagevu/nifti/train/"
LABEL_DIR = "/home/local/VANDERBILT/litz/data/imagevu/lobe/uniform_lvlsetseg/"
LOG_DIR = "/home/local/VANDERBILT/litz/github/MASILab/lobe_seg/logs"
# EXAMPLE_FNAME = "00000808time20180312.nii.gz"

def main():
    EPOCHS = 1
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 1

    # resize to 192,192,128
    transform = transforms.Compose([
        transforms.Resize((192,192,128), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])
    dataset = NiftiDataset(dataset_dir=TRAIN_DIR, label_dir=LABEL_DIR, sample_size=2, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    # Print dataset stats
    print(f"Dataset len = {len(dataset)}")

    # Logging

    # Train
    model = Unet3d()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        for data, label in loader:
            data = Variable(data.cuda())
            label = Variable(label.cuda())

            pred = model(data)
            loss = cross_entropy_loss(pred, label)

            # calculate dice
            # pred = pred.squeeze().data.cpu().numpy()
            # label = label.squeeze().cpu().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, loss = {loss.item() / BATCH_SIZE}, dice = {dice}")


if __name__ == '__main__':
    main()