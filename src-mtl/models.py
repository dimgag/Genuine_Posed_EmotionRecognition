import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict


class HydraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(pretrained=True)
        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Identity()
        self.net.fc1 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('final', nn.Linear(self.n_features, 1))]))
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
        # self.n_features = self.net.fc.in_features

        self.net.fc = nn.Identity()
        self.net.fc1 = nn.Sequential(OrderedDict([('linear', nn.Linear(512,256)),('relu1', nn.ReLU()),('final', nn.Linear(256, 2)), ('sigmoid', nn.Sigmoid())])) # This should be 2 for real/fake? 
        self.net.fc2 = nn.Sequential(OrderedDict([('linear', nn.Linear(512,256)),('relu1', nn.ReLU()),('final', nn.Linear(256, 6))]))
        
    def forward(self, x):
        real_fake_head = self.net.fc1(self.net(x))
        emotion_head = self.net.fc2(self.net(x))
        return real_fake_head, emotion_head

