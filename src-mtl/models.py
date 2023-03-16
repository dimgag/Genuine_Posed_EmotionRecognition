import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict


class HydraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(pretrained=True)
        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Identity()
        self.net.fc1 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('final', nn.Linear(self.n_features, 2))]))
        self.net.fc2 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('final', nn.Linear(self.n_features, 6))]))
        
    def forward(self, x):
        real_fake_head = self.net.fc1(self.net(x))
        emotion_head = self.net.fc2(self.net(x))
        return real_fake_head, emotion_head



from facenet_pytorch import InceptionResnetV1

class ChimeraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = InceptionResnetV1(pretrained='vggface2')

        self.net.fc = nn.Identity()
        self.net.fc1 = nn.Sequential(OrderedDict([('linear', nn.Linear(512,256)),('relu1', nn.ReLU()),('final', nn.Linear(256, 2))]))
        self.net.fc2 = nn.Sequential(OrderedDict([('linear', nn.Linear(512,256)),('relu1', nn.ReLU()),('final', nn.Linear(256, 6))]))
        
    def forward(self, x):
        real_fake_head = self.net.fc1(self.net(x))
        emotion_head = self.net.fc2(self.net(x))
        return real_fake_head, emotion_head


###############
class ChimeraNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = InceptionResnetV1(pretrained='vggface2')

        self.net.conv2d_1a = nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.net.conv2d_2a = nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False)
        self.net.conv2d_2b = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool_3a = nn.MaxPool2d(kernel_size=3, stride=2)
        self.net.conv2d_3b = nn.Conv2d(256, 384, kernel_size=3, stride=2, bias=False)

        self.net.fc = nn.Identity()
        self.net.fc1 = nn.Sequential(
            nn.Linear(1792, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.net.fc2 = nn.Sequential(
            nn.Linear(1792, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        real_fake_head = self.net.fc1(self.net(x))
        emotion_head = self.net.fc2(self.net(x))
        return real_fake_head, emotion_head
