import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# My modules
from dataset import MultiTaskDataset
from model import MultiTaskTimeSformer
from train import train, validate

def main():
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                # transforms.RandomHorizontalFlip(p=0.5), #Data Augmentation
                # transforms.RandomRotation(35), #Data Augmentation
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4270, 0.3508, 0.2971], std=[0.1844, 0.1809, 0.1545])
                ])

    # Define the dataset
    dataset = MultiTaskDataset(data_path='data_temporal', transform=transform)

    # Split the dataset into train and test sets
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

    # Batch size
    batch_size = 32

    # create the data loaders for train and test sets
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Create the model
    num_classes = 6
    num_tasks = 2

    # Define the model
    model = MultiTaskTimeSformer(num_classes=num_classes, num_tasks=num_tasks)

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # ADAMW optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001) # Might be better than SGD for this model.

    # SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # define the number of epochs
    num_epochs = 10

    # define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # move the model to the device
    model.to(device)

    # train the model
    # for epoch in range(num_epochs):
        # ... train the model ...
        # To be to continued...

if __name__ == '__main__':
    # main()
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                # transforms.RandomHorizontalFlip(p=0.5), #Data Augmentation
                # transforms.RandomRotation(35), #Data Augmentation
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4270, 0.3508, 0.2971], std=[0.1844, 0.1809, 0.1545])
                ])

    # Define the dataset
    dataset = MultiTaskDataset(data_path='data_temporal', transform=transform)

    # Split the dataset into train and test sets
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

    # Batch size
    batch_size = 32

    # create the data loaders for train and test sets
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Create the model
    num_classes = 6
    num_tasks = 2
