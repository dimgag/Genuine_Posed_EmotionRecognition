'''
MAIN IDEA:

I want to implement a Multi-Task Learning Transformer model with temporal information for Image Classification.
The model should be able to predict the following tasks:
Classify the emotion of a person in a given video
Classify if the emotion is real or fake

For that purpose I have the following dataset:
data_temporal /
  train_root/
           /fake_angry
           /fake_contempt
           /fake_disgust
           /fake_happy
           /fake_sad
           /fake_surprise
           /real_angry
           /real_contempt
           /real_disgust
           /real_happy
           /real_sad
           /real_surprise
  val_root/
           /fake_angry
           /fake_contempt
           /fake_disgust
           /fake_happy
           /fake_sad
           /fake_surprise
           /real_angry
           /real_contempt
           /real_disgust
           /real_happy
           /real_sad
           /real_surprise
Where each folder contains videos.

'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
'''
class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(self.root_dir)

    def __len__(self):
        return sum([len(os.listdir(os.path.join(self.root_dir, c))) for c in self.classes])

    def __getitem__(self, idx):
        class_folders = os.listdir(self.root_dir)
        class_idx = np.random.randint(0, len(class_folders))
        class_folder = class_folders[class_idx]
        class_videos = os.listdir(os.path.join(self.root_dir, class_folder))
        video_idx = np.random.randint(0, len(class_videos))
        video_path = os.path.join(self.root_dir, class_folder, class_videos[video_idx])
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        label1 = class_idx
        label2 = int(class_folder.split("_")[0] == "real")
        return torch.stack(frames), label1, label2


'''




class MultiTaskTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim1, output_dim2, num_heads, num_layers):
        super(MultiTaskTransformer, self).__init__()
        self.d_model = input_dim
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(input_dim, num_heads, hidden_dim), num_layers)
        self.fc1 = nn.Linear(input_dim, output_dim1)
        self.fc2 = nn.Linear(input_dim, output_dim2)

    def forward(self, x):
        x = self.transformer(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return x1, x2

# example usage
mtl_model = MultiTaskTransformer(3, 16, 6, 2, 3, 4)


# input_tensor = torch.randn(10, 20, 3) # batch size x sequence length x input dimension
# output1, output2 = mtl_model(input_tensor)
# print(output1.shape, output2.shape)

# Find trainable parameters
total_params = sum(p.numel() for p in mtl_model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in mtl_model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')




'''
# Hyperparameters
input_dim = 3 # RGB channels
hidden_dim = 128
output_dim1 = 6 # 6 classes for task 1
output_dim2 = 2 # 2 classes for task 2
num_heads = 8
num_layers = 4
batch_size = 32
lr = 0.001
num_epochs = 10

# Data
train_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = VideoDataset("data_temporal/train_root", transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
val_dataset = VideoDataset("data_temporal/val_root", transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, loss and optimizer
model = MultiTaskTransformer(input_dim, hidden_dim, output_dim1, output_dim2, num_heads, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train the model
for epoch in range(num_epochs):
    running_loss1, running_loss2 = 0.0, 0.0
    for inputs, labels1, labels2 in train_loader:
        optimizer.zero_grad()
        outputs1, outputs2 = model(inputs)
        loss1 = criterion(outputs1, labels1)
        loss2 = criterion(outputs2, labels2)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
    print(f"Epoch {epoch+1}: Loss1={running_loss1/len(train_loader):.4f}, Loss2={running_loss2/len(train_loader):.4f}")
    # Validate the model
    correct1, total1, correct2, total2 = 0, 0, 0, 0
    with torch.no_grad():
        for inputs, labels1, labels2 in val_loader:
            outputs1, outputs2 = model(inputs)
            _, predicted1 = torch.max(outputs1.data, 1)
            _, predicted2 = torch.max(outputs2.data, 1)
            total1 += labels1.size(0)
            total2 += labels2.size(0)
            correct1 += (predicted1 == labels1).sum().item()
            correct2 += (predicted2 == labels2).sum().item()
    print(f"Accuracy1={correct1/total1:.4f}, Accuracy2={correct2/total2:.4f}")

'''