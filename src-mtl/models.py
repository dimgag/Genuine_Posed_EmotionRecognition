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






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Add to main.py
# #
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# net = HydraNet(pretrained=True)
# model = HydraNet(net).to(device)
# emotion_loss = nn.CrossEntropyLoss() # Includes Softmax
# real_fake_loss = nn.BCELoss() # Doesn't include Softmax
from facenet_pytorch import InceptionResnetV1

class HydraFaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = InceptionResnetV1(pretrained='vggface2')
        self.n_features = self.net.fc.in_features

        self.net.fc = nn.Identity()
        self.net.fc1 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('final', nn.Linear(self.n_features, 1))]))
        self.net.fc2 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('final', nn.Linear(self.n_features, 6))]))
        
    def forward(self, x):
        real_fake_head = self.net.fc1(self.net(x))
        emotion_head = self.net.fc2(self.net(x))
        return real_fake_head, emotion_head

# To be continued...