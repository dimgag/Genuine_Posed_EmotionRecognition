import torch
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import HydraNet, ChimeraNet
from dataset import SASEFE_MTL, SASEFE_MTL_TEST
matplotlib.style.use('ggplot')


def CM(model, test_loader):

    # 
    counter = 0
    
    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        counter += 1
        inputs = data["image"].to(device)
        real_fake_label = data["real_fake"].to(device) 
        emotion_label = data["emotion"].to(device)

        real_fake_output, emotion_output = model(inputs)
        
        
        # So I have emotion_output = prediction and emotion_label = GT
        # generate Confussion matrix for real_fake: 
    
    real_fake_output = real_fake_output.cpu()
    real_fake_output = real_fake_output.detach().numpy()
    real_fake_output = np.round(real_fake_output)
    # emotion_output = np.argmax(emotion_output, axis=0)
    
    
    real_fake_cm = confusion_matrix(real_fake_classes, real_fake_output)
    print(real_fake_cm)




if __name__ == "__main__":
    print("Evaluate model")
    torch.cuda.empty_cache()

    # Load the model.pth - change the path to the model you want to evaluate
    path = 'experiments/exp1-chimeranet/model.pth'

    # Device configuration
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define the model class
    # loaded_model = HydraNet().to(device)
    loaded_model = ChimeraNet().to(device)

    # Define the optimizer
    optimizer = torch.optim.SGD(loaded_model.parameters(), lr=1e-4, momentum=0.09)

    # Define the loss functions.
    emotion_loss = nn.CrossEntropyLoss()
    real_fake_loss = nn.BCELoss()

    # Load the checkpoint
    loaded_checkpoint = torch.load(path, map_location=torch.device('cpu'))

    # Load the model state
    loaded_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
    epoch = loaded_checkpoint['epoch']

    # Load the data
    test_dir = "data_mtl/test"
    test_image_paths = os.listdir(test_dir)

    # Get the dataset class
    test_dataset = SASEFE_MTL_TEST(test_image_paths)

    # Get the dataloaders
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    
    CM(loaded_model, test_dataloader)