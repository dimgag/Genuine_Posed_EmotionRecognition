# Fine tune models

# Path: src/fine_tune.py
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.model1 import Net
from models.vggface import VGGFace
from models.vggface import VGGFace2


# Get the model
def get_model(model_name):
    if model_name == 'model1':
        model = Net()
    elif model_name == 'vggface':
        model = VGGFace()
    elif model_name == 'vggface2':
        model = VGGFace2()
    else:
        print('Model not found')
        sys.exit(1)
    return model

# Freeze the model and train the last layer
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier[-1].parameters():
        param.requires_grad = True 

    return model


def get_model_params(model):
  """Get model parameters"""
  total_params = sum(p.numel() for p in model.parameters())
  print(f"Total parameters: {total_params:,}")
  total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"Trainable parameters: {total_trainable_params:,}")


def add_classification_head(model, device, num_classes=12):
    model.classifier[3] = torch.nn.Sequential(
        torch.nn.Linear(in_features=1280, out_features=640, bias=True),
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=640, out_features=320, bias=True),
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=320, out_features=num_classes, bias=True)).to(device)

    # Print the model parameters
    print("New model parameters after adding Classification head:")
    get_model_params(model)




def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model('vggface2')
    get_model_params(model)
    
    model = freeze_model(model)
    get_model_params(model)

    print("Adding Classification Head . . .")
    add_classification_head(model, device)





if __name__ == '__main__':
    main()



