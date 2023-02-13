import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

image_size = 224
batch_size = 32
num_workers = 4

# Data directories
train_dir = "data/train"
test_dir = "data/test"

train_images = os.listdir(train_dir)
test_images = os.listdir(test_dir)


# Remove .DS_Store files
if '.DS_Store' in train_images:
    train_images.remove('.DS_Store')
if '.DS_Store' in test_images:
    test_images.remove('.DS_Store')


## Data Augmentation
# Training Data Transforms
def get_train_transform(image_size):
  train_transform = transforms.Compose([
      transforms.Resize((image_size, image_size)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomRotation(35),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.4270, 0.3508, 0.2971], std=[0.1844, 0.1809, 0.1545])
    ])
  return train_transform

# Test Data Transforms
def get_test_transform(image_size):
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4270, 0.3508, 0.2971], std=[0.1844, 0.1809, 0.1545])
    ])
    return test_transform


def get_datasets():
    """
    Function to prepare the Datasets.
    Returns the training and Test datasets along 
    with the class names.
    """
    dataset_train = datasets.ImageFolder(
        train_dir, 
        transform=(get_train_transform(image_size))
    )
    dataset_test = datasets.ImageFolder(
        test_dir, 
        transform=(get_test_transform(image_size))
    )
    return dataset_train, dataset_test, dataset_train.classes


def get_data_loaders(dataset_train, dataset_test):
    """
    Input: the training and Test data.
    Returns the training and Test data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader 


def get_mean_and_std(dataloader):
    '''Compute the mean and std value of dataset.'''
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

#  Try it out
# dataset_train, dataset_test, dataset_classes = get_datasets()
# Load the training and Test data loaders
# train_loader, test_loader = get_data_loaders(dataset_train, dataset_test)

# mean, std = get_mean_and_std(train_loader)

# print("Mean is: ", mean)
# print("\nStd is: ", std)
