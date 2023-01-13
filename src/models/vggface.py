import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGFace(nn.Module):
    def __init__(self):
        super(VGGFace, self).__init__()
        self.features = models.vgg16(pretrained=True).features
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 12), # 12 Output classes
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class VGGFace2(nn.Module):
    def __init__(self):
        super(VGGFace2, self).__init__()
        self.features = models.vgg16(weights='VGGFace2').features
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 12), # 12 Output classes
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


