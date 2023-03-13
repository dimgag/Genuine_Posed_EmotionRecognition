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

        # rf_output = torch.mean(rf_output, dim=0, keepdim=False)
        # emo_output = torch.mean(emo_output, dim=0, keepdim=False)
        # print("real/fake shape:", rf_output.shape)
        # print("emotions shape:", emo_output.shape)

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
        
        n_frames = 16 # Change this to the desired number of frames

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




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, val_loader, optimizer, epochs):
    print("Training model...")
    for epoch in range(epochs):
        model.train()
        emotion_loss = nn.CrossEntropyLoss()
        real_fake_loss = nn.CrossEntropyLoss()
        total_training_loss = 0.0
        emotion_training_acc = 0
        real_fake_training_acc = 0
        overall_training_acc = 0
        counter = 0
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()
            inputs = data["frames"].to(device)
            real_fake_label = data["rf_label"].to(device)
            emotion_label = data["emo_label"].to(device)
            
            real_fake_output, emotion_output = model(inputs)
            
            # ------------------------------------------------------------
            # Calculate the Losses
            loss_1 = emotion_loss(emotion_output, emotion_label)
            loss_2 = real_fake_loss(real_fake_output, real_fake_label)
            loss = loss_1 + loss_2
            print(loss)
            # ------------------------------------------------------------
            # Calculate Precision for emotions
            _, emo_preds = torch.max(emotion_output.data, 1)
            emo_true_positives = (emo_preds == emotion_label).sum().item()
            emo_total_predicted_positives = emotion_label.shape[0]
            precision_emotions = emo_true_positives / emo_total_predicted_positives
            # Calculate Precision for real/fake
            _, rf_preds = torch.max(real_fake_output.data, 1)
            rf_true_positives = (rf_preds == real_fake_label).sum().item()
            rf_total_predicted_positives = real_fake_label.shape[0]
            precision_real_fake = rf_true_positives / rf_total_predicted_positives
            # Calculate the combined loss
            loss_emotions = precision_emotions * loss_1
            loss_real_fake = precision_real_fake * loss_2
            loss = loss_emotions + loss_real_fake
            total_training_loss += loss.item()
            counter += 1
            # ------------------------------------------------------------
            # Calculate Accuracy for Emotions
            _, emo_preds = torch.max(emotion_output.data, 1)
            emotion_training_acc += (emo_preds == emotion_label).sum().item()
            # Calculate Accuracy for Real/Fake
            _, rf_preds = torch.max(real_fake_output.data, 1)
            real_fake_training_acc += (rf_preds == real_fake_label).sum().item()
            # Calculate overall accuracy
            overall_training_acc += (rf_preds == real_fake_label).sum().item()
            overall_training_acc += (emo_preds == emotion_label).sum().item()
            # Backpropagation
            loss.backward()
            # Update the weights
            optimizer.step()
        epoch_loss = total_training_loss / counter 
        epoch_acc_emotion = 100. * (emotion_training_acc / len(train_loader.dataset))
        epoch_acc_real_fake = 100. * (real_fake_training_acc / len(train_loader.dataset))
        overall_training_acc = 100. * (overall_training_acc / (2*len(train_loader.dataset)))
        print(f"Epoch {epoch+1} loss: {epoch_loss:.4f} | Emotion accuracy: {epoch_acc_emotion:.2f}% | Real/Fake accuracy: {epoch_acc_real_fake:.2f}% | Overall accuracy: {overall_training_acc:.2f}%")





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
    epoch_loss, epoch_acc_emotion, epoch_acc_real_fake, overall_training_acc = train(model, train_dataloader, test_dataloader, optimizer, num_epochs)




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
  

