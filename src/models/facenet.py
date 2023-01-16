from facenet_pytorch import InceptionResnetV1
import torch.nn as nn
import torch

# For a model pretrained on VGGFace2
# model = InceptionResnetV1(pretrained='vggface2').eval()

# define device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Define the model class
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
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 12), # 12 Output classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
        