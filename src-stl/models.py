import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models 
from facenet_pytorch import InceptionResnetV1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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



class EfficientNetV2M(nn.Module):
    def __init__(self):
        super(EfficientNetV2M, self).__init__()
        self.model = models.efficientnet_v2_m(pretrained=True)
        self.model.classifier = nn.Linear(1280, 12)

    def forward(self, x):
        x = self.model(x)
        return x


# VGG16 pretrained on ImageNet
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
        self.features = models.vgg16(weights=True).features
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


#  FaceNet model pretrained on VGGFace2
class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        self.model = InceptionResnetV1(
            classify=True,
            pretrained='vggface2',
            num_classes=12
        ).to(device)

    def forward(self, x):
        x = self.model(x)
        return x


class FaceNet_withClassifier(nn.Module):
    def __init__(self):
        super(FaceNet_withClassifier, self).__init__()
        self.features = InceptionResnetV1(pretrained='vggface2')
        # print(self.features)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 12) # 12 Output classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



# Vision transformer-b16 pretrained on ImageNet
class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.model = models.vit_b_16(pretrained=True)
        self.model.classifier = nn.Linear(768, 12)

    def forward(self, x):
        x = self.model(x)
        return x
