# HydraNet - Multi Task approach for Real/Fake and Emotion Classification
# The dataset is a set of images, ana the names of the images give the labels.
# For example the image "real_happy_1.jpg" is a real image of a happy person.

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

# Import modues
from dataset import SASEFE_MTL



def main():
    # Data directory
    train_dir = "data_mtl/train"
    test_dir = "data_mtl/test"
    test_image_paths = os.listdir(test_dir)
    train_image_paths = os.listdir(train_dir)

    # Get the dataset class
    train_dataset = SASEFE_MTL(train_image_paths)
    test_dataset = SASEFE_MTL(test_image_paths)

    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)


    print("Number of training images: ", len(train_dataset))
    print("Number of test images: ", len(test_dataset))



if __name__ == "__main__":
    main()
