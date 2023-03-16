import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class MyNetwork(nn.Module):
    '''Not giving more than 8% val_acc'''
    def __init__(self, num_classes):
        super(MyNetwork, self).__init__()
        self.resnet = InceptionResnetV1(pretrained='vggface2', classify=False)

        self.fc1 = nn.Linear(1, 512)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, num_classes)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        batch_size, num_channels, num_frames, height, width = x.shape
        x = x.reshape(-1, num_channels, height, width)  # Flatten temporal dimension into batch dimension
        x = self.resnet(x)
        x = x.reshape(batch_size, num_frames, -1)
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    


class EmotionRecognitionModel2(nn.Module):
    '''Not giving more than 8% val_acc'''
    def __init__(self, num_classes):
        super(EmotionRecognitionModel2, self).__init__()
        self.inception_resnet = InceptionResnetV1(pretrained='vggface2', classify=False)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=1, out_features=num_classes)

    def forward(self, x):
        batch_size, num_channels, num_frames, height, width = x.shape
        x = x.reshape(-1, num_channels, height, width)  # Flatten temporal dimension into batch dimension
        x = self.inception_resnet(x)
        x = x.reshape(batch_size, num_frames, -1)  # Reshape feature maps back to 3D tensor
        x = x.permute(0, 2, 1)  # Permute tensor to have shape (batch_size, num_features, num_frames)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
