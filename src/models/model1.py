import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 12)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 53 * 53)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# Make a class for efficientnet_v2_m

class EfficientNetV2M(nn.Module):
    def __init__(self):
        super(EfficientNetV2M, self).__init__()
        self.model = models.efficientnet_v2(pretrained=True)
        self.model.classifier = nn.Linear(1280, 12)

    def forward(self, x):
        x = self.model(x)
        return x
