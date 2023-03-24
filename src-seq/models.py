import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

    
class MODEL_3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(MODEL_3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.bn3 = nn.BatchNorm3d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(125440, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x




# LSTM Model - simple LSTM model    -> It started working!!! (Slow)
class MODEL_LSTM(nn.Module):
    def __init__(self, num_classes):
        super(MODEL_LSTM, self).__init__()
        self.lstm1 = nn.LSTM(1003520, 256, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(256, 512)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size, timesteps, -1)
        x, _ = self.lstm1(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
    
    

# GRU Model - simple GRU model
class MODEL_GRU(nn.Module):
    def __init__(self, num_classes):
        super(MODEL_GRU, self).__init__()
        self.gru1 = nn.GRU(input_size=3010560, hidden_size=256, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Reshape the input to (batch_size, sequence_length, input_size)
        x = x.reshape(x.size(0), 1, -1)
        # Pass through the GRU layer
        x, _ = self.gru1(x)
        # Flatten the output and pass through fully connected layers
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x




    

# X3D Model from X3D networks pretrained on the Kinetics 400 dataset 
# PyTorchVideo: https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/
# Choose the `x3d_s` model
# model_name = 'x3d_s'
# model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)

class MODEL_X3D(nn.Module):
    def __init__(self, num_classes):
        super(MODEL_X3D, self).__init__()
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


# 3D RESNET pretrained on Kinetics dataset.
# Try this one: 3D RESNET https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/
# model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

class MODEL_3D_RESNET(nn.Module):
    def __init__(self, num_classes):
        super(MODEL_3D_RESNET, self).__init__()
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


# Efficient X3D model pretrained on Kinetics dataset.

class MODEL_efficient_x3d(nn.Module):
    def __init__(self, num_classes):
        super(MODEL_efficient_x3d, self).__init__()
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'efficient_x3d_s', pretrained=True)
        self.model.fc = nn.Linear(400, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
