import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings
warnings.filterwarnings("ignore")


## Packages actually used in this file
import os

# Data directory:
data_dir = "/Users/dim__gag/Desktop/SASE-FE/FakeTrue_DB"

# Frames Directories:
frames_dir = '/Users/dim__gag/Desktop/SASE-FE/frames'
real_dir = '/Users/dim__gag/Desktop/SASE-FE/frames/real/'
fake_dir = '/Users/dim__gag/Desktop/SASE-FE/frames/fake/'

# Two main labels:
main_labels = os.listdir(frames_dir)
main_labels.remove('.DS_Store')
print("Main labels:", main_labels)

# Real emotions labels:
real_emotions_labels = os.listdir(real_dir)
real_emotions_labels.remove('.DS_Store')
print("\nReal emotions labels:", real_emotions_labels)

# Fake emotions labels:
fake_emotions_labels = os.listdir(real_dir)
fake_emotions_labels.remove('.DS_Store')
print("\nFake emotions labels:", fake_emotions_labels)





# Split data into training and validation sets
from sklearn.model_selection import train_test_split
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Convert data to torch tensors
train_data = torch.from_numpy(train_data)
val_data = torch.from_numpy(val_data)
train_labels = torch.from_numpy(train_labels)
val_labels = torch.from_numpy(val_labels)

# Create a dataset class
class VideoDataset(Dataset):
    """Video dataset."""

    def __init__(self, data, labels, transform=None):
        """
        Args:
            data (numpy array): A numpy array of video data.
            labels (numpy array): A numpy array of video labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

# Create a dataset object
train_dataset = VideoDataset(train_data, train_labels)
val_dataset = VideoDataset(val_data, val_labels)

# Create a dataloader object
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


