import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class MyNetwork(nn.Module):
    def __init__(self, num_classes):
        super(MyNetwork, self).__init__()
        self.resnet = InceptionResnetV1(pretrained='vggface2', classify=False)

        self.fc1 = nn.Linear(512, 512)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    


class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(EmotionRecognitionModel, self).__init__()
        self.inception_resnet = InceptionResnetV1(pretrained='vggface2', classify=False)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.inception_resnet(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x









class Snowbaby(nn.Module):
    def __init__(self, num_classes):
        super(Snowbaby, self).__init__()
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
        self.fc1 = nn.Linear(512, 512) # change the input size to 512
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, num_classes)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

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
        # extract features using the resnet model
        with torch.no_grad():
            features = self.resnet(x)
        # concatenate the extracted features with the output of the existing model
        x = torch.cat([x, features], dim=1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x

