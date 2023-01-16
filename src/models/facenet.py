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

