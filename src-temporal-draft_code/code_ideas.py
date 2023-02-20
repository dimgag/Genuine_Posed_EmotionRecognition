# Original repo: https://github.com/facebookresearch/TimeSformer
# Use this library: `from timesformer_pytorch import TimeSformer`  
# Repo: https://github.com/lucidrains/TimeSformer-pytorch
# `$ pip install timesformer-pytorch`


#############################################################################
# dataset preparation
import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# Define the path to your dataset
dataset_path = "/path/to/dataset"

# Define the path to the output directory for the extracted frames
frames_path = "/path/to/frames"

# Define the path to the output directory for the metadata files
metadata_path = "/path/to/metadata"

# Define the size of the output frames
frame_size = (224, 224)

# Define the train/val/test split ratios
train_split = 0.7
val_split = 0.2
test_split = 0.1

# Define the class labels
class_labels = ["class1", "class2", "class3", ...]

# Create the output directories
os.makedirs(frames_path, exist_ok=True)
os.makedirs(metadata_path, exist_ok=True)

# Initialize the metadata dictionary
metadata = {
    "train": [],
    "val": [],
    "test": []
}

# Loop over the videos and extract the frames
for class_label in class_labels:
    class_path = os.path.join(dataset_path, class_label)
    video_files = os.listdir(class_path)
    num_videos = len(video_files)

    # Shuffle the video files
    np.random.shuffle(video_files)

    # Split the videos into train, val, and test sets
    train_videos = video_files[:int(train_split * num_videos)]
    val_videos = video_files[int(train_split * num_videos):int((train_split + val_split) * num_videos)]
    test_videos = video_files[int((train_split + val_split) * num_videos):]

    # Loop over the train, val, and test sets
    for split, videos in zip(["train", "val", "test"], [train_videos, val_videos, test_videos]):
        for video_file in tqdm(videos, desc=f"Processing {split} videos"):
            # Create a directory for the video frames
            video_id = os.path.splitext(video_file)[0]
            video_frames_path = os.path.join(frames_path, video_id)
            os.makedirs(video_frames_path, exist_ok=True)

            # Read the video and extract the frames
            video_path = os.path.join(class_path, video_file)
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for frame_id in range(frame_count):
                success, frame = cap.read()
                if not success:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, frame_size)
                frame_path = os.path.join(video_frames_path, f"{frame_id}.jpg")
                cv2.imwrite(frame_path, frame)

            # Add the video metadata to the corresponding split in the metadata dictionary
            video_metadata = {
                "video_path": os.path.join(video_frames_path, "*.jpg"),
                "label": class_label
            }
            metadata[split].append(video_metadata)

# Write the metadata dictionary to file
with open(os.path.join(metadata_path, "metadata.json"), "w") as f:
    json.dump(metadata, f)

# In this code, you'll need to replace the placeholders "/path/to/dataset", "/path/to/frames",
#  and "/path/to/metadata" with the actual paths to your dataset and output directories. 
# You'll also need to specify the size of the output frames, the train/val/test split ratios, and the class labels.



#############################################################################
# Example for training - NOTE not a multi-task learning approach!
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from timesformer_pytorch import TimeSformer

# Define the path to the metadata file
metadata_path = "/path/to/metadata/metadata.json"

# Define the batch size and number of workers for the data loader
batch_size = 8
num_workers = 4

# Define the number of classes
num_classes = 3

# Define the number of epochs to train
num_epochs = 10

# Define the learning rate
learning_rate = 0.001

# Define the device to train on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transforms to apply to the frames
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load the metadata file
with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Create the datasets and data loaders
datasets = {
    split: datasets.DatasetFolder(
        root="",
        loader=lambda x: Image.open(x),
        extensions=("jpg",),
        transform=transforms,
        data=metadata[split]
    )
    for split in ["train", "val", "test"]
}
dataloaders = {
    split: DataLoader(
        dataset=datasets[split],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    for split in ["train", "val", "test"]
}

# Create the model and optimizer
model = TimeSformer(
    num_classes=num_classes,
    num_frames=16,
    image_size=224,
    patch_size=16,
    hidden_dim=256,
    num_heads=8,
    num_layers=4
).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Train the model
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    for split in ["train", "val", "test"]:
        if split == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(dataloaders[split], desc=f"{split} batch"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(split == "train"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if split == "train":
                    loss.backward()
                    optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(datasets[split])
        epoch_acc = running_corrects.double() / len(datasets[split])

        print(f"{split} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

# In this code, you'll need to replace the placeholders "/path/to/metadata/metadata.json" with the actual path to your metadata file. 
#############################################################################

# Dataset class idea for creating TimeSformer-like Multi-Task learning dataset


import torch
import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class SASEFE_MTL(Dataset):
    def __init__(self, metadata_path, num_classes_task1, num_classes_task2):
        # Define the transforms to apply to the frames
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4270, 0.3508, 0.2971], std=[0.1844, 0.1809, 0.1545])
        ])

        # Load the metadata file
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Set Inputs and Labels
        self.image_paths = metadata["image_paths"]
        self.real_fakes = metadata["real_fakes"]
        self.emotions = metadata["emotions"]
        self.num_classes_task1 = num_classes_task1
        self.num_classes_task2 = num_classes_task2

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Load the image and apply transforms
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = self.transform(image)

        # Get the labels for task 1 and task 2
        real_fake_label = self.real_fakes[index]
        emotion_label = self.emotions[index]

        # Convert the labels to one-hot encoding
        real_fake_onehot = torch.zeros(self.num_classes_task1)
        real_fake_onehot[real_fake_label] = 1
        emotion_onehot = torch.zeros(self.num_classes_task2)
        emotion_onehot[emotion_label] = 1

        # Return the inputs and labels
        return image, (real_fake_onehot, emotion_onehot)


# In this modified SASEFE_MTL class, 
# the metadata_path argument is the path to the metadata file containing the paths to the images, the real/fake labels, and the emotion labels. 
# The num_classes_task1 and num_classes_task2 arguments are the number of classes for each task.

# The __getitem__ method returns a tuple of the input image and a tuple of one-hot encoded labels for each task. 
# The first element of the label tuple is the one-hot encoded label for task 1 (real/fake), and the second element is the one-hot encoded label for task 2 (emotion). 
# This format is suitable for training the TimeSformer in a multi-task learning setting.
