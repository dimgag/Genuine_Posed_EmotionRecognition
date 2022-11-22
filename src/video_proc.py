
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


plt.ion()   # interactive mode

# Load data
data_dir = 'data/'
data = np.load(data_dir + 'data.npy')
labels = np.load(data_dir + 'labels.npy')

# Load a video dataset from the web
import urllib.request
import os
import zipfile

# Download the data
data_url = 'https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5bf60c3a_data/data.zip'
data_dir = 'data/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.isfile(data_dir + 'data.zip'):
    print('Downloading data ...')
    urllib.request.urlretrieve(data_url, data_dir + 'data.zip')
    print('Done!')
# Unzip the data
if not os.path.isfile(data_dir + 'data.npy'):
    print('Unzipping data ...')
    with zipfile.ZipFile(data_dir + 'data.zip', 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print('Done!')

# Load data
data = np.load(data_dir + 'data.npy')
labels = np.load(data_dir + 'labels.npy')





# Split data into training and validation sets
from sklearn.model_selection import train_test_split
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create a dataset class
class VideoDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, labels, transform=None):
        """
        Args:
            data (numpy array): Numpy array of video data.
            labels (numpy array): Numpy array of video labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'labels': self.labels[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Create a transform class
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, labels = sample['data'], sample['labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        data = data.transpose((0, 3, 1, 2))
        return {'data': torch.from_numpy(data),
                'labels': torch.from_numpy(labels)}

# Create a dataset
train_dataset = VideoDataset(train_data, train_labels, transform=transforms.Compose([ToTensor()]))  
val_dataset = VideoDataset(val_data, val_labels, transform=transforms.Compose([ToTensor()]))
print('Training data size: ', len(train_dataset))
print('Validation data size: ', len(val_dataset))

# Create a dataloader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)

# Create a Xception model in pytorch
# Path: src/model.py
# Build a cnn model in pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F

class Xception(nn.Module):
    def __init__(self):
        super(Xception, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 728, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(728)
        self.conv8 = nn.Conv2d(728, 728, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(728)
        self.conv9 = nn.Conv2d(728, 728, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(728)
        self.conv10 = nn.Conv2d(728, 728, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(728)
        self.conv11 = nn.Conv2d(728, 728, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(728)
        self.conv12 = nn.Conv2d(728, 728, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(728)
        self.conv13 = nn.Conv2d(728, 1024, 3, stride=2, padding=1)
        self.bn13 = nn.BatchNorm2d(1024)
        self.conv14 = nn.Conv2d(1024, 1536, 3, padding=1)
        self.bn

