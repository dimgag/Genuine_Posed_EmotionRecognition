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
from models.facenet import FaceNet_withClassifier
from models.facenet import FaceNet

def get_model(model_name):
    if model_name == 'model1':
        model = Net()
    elif model_name == 'vggface':
        model = VGGFace()
    elif model_name == 'vggface2':
        model = VGGFace2()
    elif model_name == 'facenet':
        model = FaceNet_withClassifier()
    else:
        print('Model not found')
        sys.exit(1)
    return model


def get_model_params(model):
  """Get model parameters"""
  total_params = sum(p.numel() for p in model.parameters())
  print(f"Total parameters: {total_params:,}")
  total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"Trainable parameters: {total_trainable_params:,}")




# Freeze the model parameters except classification head

def freeze_model(model):
    print("-"*50)
    print("\nModel parameters before freezing: ", get_model_params(model))

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True 

    print("-"*50)
    print("\nModel parameters after freezing the base model: ", get_model_params(model))

    return model


def add_classification_head(model, device, num_classes=12):
    model.classifier[3] = torch.nn.Sequential(
        torch.nn.Linear(in_features=256, out_features=128, bias=True),
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=128, out_features=64, bias=True),
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=64, out_features=num_classes, bias=True)).to(device)
    
    print("\nModel parameters after adding new classification head: ", get_model_params(model))

    return model




def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model('facenet')
    model = freeze_model(model)
    model = add_classification_head(model, device)






if __name__ == '__main__':
    main()



