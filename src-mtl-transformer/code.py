import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from PIL import Image



# define a multi-task temporal transformer model that gets as input torch.Size([1, 17, 256, 256, 3]), where 1 is the batch size, 17 is the number of frames, 256 is the height and width of the frames, and 3 is the number of channels (RGB)
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalTransformer(nn.Module):
    def __init__(self, num_classes1, num_classes2):
        super(TemporalTransformer, self).__init__()
        
        # 3D CNN to extract features from the video frames
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Multi-head self-attention mechanism to capture temporal dependencies
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes1)
        self.fc3 = nn.Linear(128, num_classes2)

    def forward(self, x):
        # Reshape to: torch.Size([3, 17, 256, 256])
        x = x.permute(4, 0, 1, 2, 3).reshape(-1, 17, 256, 256)
        x = x.unsqueeze(0)  # Add batch dimension
        # Normalize and make them scalar type Byte 
        x = x.float()

        # Convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Reshape to: torch.Size([17, 128, 16, 16])
        x = x.view(x.size(0), -1, x.size(3), x.size(4))

        # Multi-head self-attention
        #  reshape to (1, 2176, 16*16)
        x = x.view(1, x.size(0), x.size(1), -1).reshape(-1, 2176, 128)

        # Multi-head self-attention
        x, _ = self.attention(x, x, x) 
        x = x.view(-1, 128 * 16 * 16)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout1(x)

        rf_output = self.fc2(x)
        emo_output = self.fc3(x)

        return rf_output, emo_output






def collate_fn(batch):
    frames = [sample['frames'] for sample in batch]
    rf_labels = [sample['rf_label'] for sample in batch]
    emo_labels = [sample['emo_label'] for sample in batch]

    # Get the maximum size of the tensors in the batch
    max_size = tuple(max(sample.size(i) for sample in frames) for i in range(1, len(frames[0].size())))

    # Pad each tensor in the batch to the maximum size
    padded_frames = []
    for sample in frames:
        # Compute the amount of padding needed
        pad = [0] * (len(sample.shape) - 1) * 2  # Pad on all dimensions except the first (batch) dimension
        for i in range(len(sample.shape) - 1):
            pad[i * 2 + 1] = max_size[i] - sample.shape[i + 1]  # Pad on the second dimension
        pad = tuple(pad)

        # Pad the tensor
        padded_frames.append(torch.nn.functional.pad(sample, pad))

    # Stack the tensors
    frames = torch.stack(padded_frames, dim=0)
    return {'frames': frames, 'rf_label': rf_labels, 'emo_label': emo_labels}






class MTL_VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Get the list of class folders
        self.class_folders = os.listdir(self.root_dir)

        # Initialize the lists of classes for each task
        self.rf_classes = ['fake', 'real']
        self.emo_classes = ['angry', 'contempt', 'disgust', 'happy', 'sad', 'surprise']

    def __len__(self):
        # Count the total number of videos in the dataset
        return sum([len(os.listdir(os.path.join(self.root_dir, c))) for c in self.class_folders])

    def __getitem__(self, idx):
        # Select a random video and its corresponding class folder
        video_folder_idx = np.random.randint(len(self.class_folders))
        video_folder = self.class_folders[video_folder_idx]
        video_path = os.path.join(self.root_dir, video_folder)
        video_files = os.listdir(video_path)
        video_file_idx = np.random.randint(len(video_files))
        video_file = video_files[video_file_idx]

        # Load the video frames
        cap = cv2.VideoCapture(os.path.join(video_path, video_file))
        frames = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        n_frames = 17 # Change this to the desired number of frames

        if total_frames < n_frames:
            # Skip videos with fewer frames than the desired number
            return self.__getitem__(np.random.randint(self.__len__()))

        step_size = total_frames // n_frames
        
        rf_label = self.rf_classes.index(video_folder.split('_')[0])
        emo_label = self.emo_classes.index(video_folder.split('_')[1])

        for i in range(0, total_frames, step_size):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
            
            # if frame none empty do this:
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(frame, (25,25), 0)
                face = face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x,y,w,h) in face:
                    frame = frame[y:y+h, x:x+w]
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (256, 256)) # Resize the frame

                if self.transform:
                    frame = self.transform(frame) # Convert the numpy array to tensor

                frames.append(frame)

        # Get the labels for the two tasks
        # rf_label = self.rf_classes.index(video_folder.split("_")[0])
        # emo_label = self.emo_classes.index(video_folder.split("_")[1])
        
        # Repear the labels for all the frames
        rf_label = torch.tensor(rf_label).repeat(n_frames)
        emo_label = torch.tensor(emo_label).repeat(n_frames)


        frames = [torch.from_numpy(frame) for frame in frames]

        
        # Pad and stack the frames
        max_size = tuple(max(sample.size(i) for sample in frames) for i in range(1, len(frames[0].size())))
        padded_frames = [torch.nn.functional.pad(sample, (0, max_size[-1] - sample.shape[-1], 0, max_size[-2] - sample.shape[-2])) for sample in frames]
        frames = torch.stack(padded_frames, dim=0)

        sample = {'frames': frames, 'rf_label': rf_label, 'emo_label': emo_label}

        return sample





class MyTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, frame):
        # Convert NumPy array to PIL Image
        frame = Image.fromarray(frame)
        # Apply transformations
        frame = transforms.Resize(self.size)(frame)
        
        # Add more transformations here if needed
        
        # Convert PIL Image back to NumPy array
        frame = np.array(frame)
        return frame




def load_mtl_dataset(data_dir, batch_size):
    # define the transform
    transform = MyTransform((256, 256))

    # define the train and test datasets
    train_dataset = MTL_VideoDataset(os.path.join(data_dir, 'train_root'), transform=transform)
    test_dataset = MTL_VideoDataset(os.path.join(data_dir, 'val_root'), transform=transform)

    # define the train and test dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

    return train_dataloader, test_dataloader





def train(model, dataset, collate_fn, criterion1, criterion2, optimizer, scheduler=None, num_epochs=10, batch_size=8, device=None):
    # Set the device
    model = model.to(device)

    # Create the data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize the losses and accuracies
    train_losses = []
    train_rf_accs = []
    train_emo_accs = []

    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        # Initialize the running losses and accuracies
        running_loss = 0.0
        running_rf_corrects = 0
        running_emo_corrects = 0
        running_total = 0

        # Set the model to training mode
        model.train()

        # Iterate over the batches
        for batch in dataset:
            # Get the inputs and labels
            inputs = batch['frames'].to(device)
            # rf_labels = torch.tensor(batch['rf_label'], dtype=torch.long).reshape(-1, 1).to(device)
            rf_labels = torch.cat(batch['rf_label'], dim=0).to(device)

            # emo_labels = torch.tensor(batch['emo_label'], dtype=torch.long).reshape(-1, 1).to(device)
            emo_labels = torch.cat(batch['emo_label'], dim=0).to(device)


            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            rf_outputs, emo_outputs = model(inputs)
            rf_loss = criterion1(rf_outputs, rf_labels)
            emo_loss = criterion2(emo_outputs, emo_labels)
            loss = rf_loss + emo_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update the running loss and accuracies
            running_loss += loss.item() * inputs.size(0)
            running_rf_corrects += torch.sum(torch.argmax(rf_outputs, dim=1) == rf_labels.view(-1)).item()
            running_emo_corrects += torch.sum(torch.argmax(emo_outputs, dim=1) == emo_labels.view(-1)).item()
            running_total += inputs.size(0)

        # Compute the epoch loss and accuracy
        epoch_loss = running_loss / running_total
        epoch_rf_acc = running_rf_corrects / running_total
        epoch_emo_acc = running_emo_corrects / running_total

        # Append the epoch loss and accuracy to the lists
        train_losses.append(epoch_loss)
        train_rf_accs.append(epoch_rf_acc)
        train_emo_accs.append(epoch_emo_acc)

        # Print the epoch loss and accuracy
        print('Epoch [{}/{}], Loss: {:.4f}, RF Acc: {:.4f}, Emo Acc: {:.4f}'.format(epoch+1, num_epochs, epoch_loss, epoch_rf_acc, epoch_emo_acc))

        # Adjust the learning rate
        if scheduler:
            scheduler.step()

    # Return the trained model and the training losses and accuracies
    return model, train_losses, train_rf_accs, train_emo_accs





# Main function to run the code from the terminal
def main():
    train_dataloader, test_dataloader = load_mtl_dataset('data_temporal', batch_size=batch_size)
    print("---------------------------------------------------")
    print("Number of training batches: ", len(train_dataloader))
    print("Number of test batches: ", len(test_dataloader))
    print("---------------------------------------------------")
    # Model, loss and optimizer
    model = TemporalTransformer(num_classes1=2, num_classes2=6)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model, train_losses, train_rf_accs, train_emo_accs = train(model, train_dataloader, collate_fn, criterion1, criterion2, optimizer, scheduler=None, num_epochs=10, batch_size=8, device=device)





if __name__ == '__main__':
  # Set the Hyperparameters
  input_dim = 3 # RGB channels
  hidden_dim = 128
  output_dim1 = 2 # 2 classes for task 1
  output_dim2 = 6 # 6 classes for task 2
  num_heads = 3
  num_layers = 4
  batch_size = 1
  lr = 0.001
  num_epochs = 10

  main()
  

