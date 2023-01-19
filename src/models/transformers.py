import torch
import torch.nn as nn
from torchvision import models


class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.model = models.vit_b_16(pretrained=True)
        self.model.classifier = nn.Linear(768, 12)

    def forward(self, x):
        x = self.model(x)
        return x
