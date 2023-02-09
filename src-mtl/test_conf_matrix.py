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


def CM(net, test_loader):
    y_pred = []
    y_true = []
    y_pred_real_fake = []
    y_pred_emotion = []
    y_true_realfake = []
    y_true_emotions = []
    counter = 0
    
    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        counter += 1
        inputs = data["image"].to(device)
        real_fake_label = data["real_fake"].to(device) 
        emotion_label = data["emotion"].to(device)

        real_fake_outputs, emotion_output = net(inputs) # Add inputs.cuda for GPU Usage

        real_fake_outputs = (torch.max(torch.exp(real_fake_outputs), 1)[1]).data.cpu().numpy()
        y_pred_real_fake.extend(real_fake_outputs)

        emotion_output = (torch.max(torch.exp(emotion_output), 1)[1]).data.cpu().numpy()
        y_pred_emotion.extend(emotion_output)

        output = net(inputs.cuda())
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
		
        y_true_realfake.extend(real_fake_label)
        y_true_emotions.extend(emotion_label)
        
        y_pred.extend(output) 

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)

    # Generate CM for real/fake
    real_fake_classes = real_fake_label
    cf_matrix = confusion_matrix(y_true_realfake, y_pred_real_fake)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix)*10, index = [i for i in real_fake_classes],
            columns = [i for i in real_fake_classes])

    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('CM_rf.png')

    # Generate CM for emotions
    emotion_classes = emotion_label
    cf_matrix = confusion_matrix(y_true_emotions, y_pred_emotion)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix)*10, index = [i for i in emotion_classes],
            columns = [i for i in emotion_classes])

    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('CM_emo.png')



if __name__ == "__main__":
    print("Evaluate model")
    torch.cuda.empty_cache()

    # Load the model.pth - change the path to the model you want to evaluate
    path = 'experiments/exp1-chimeranet/model.pth'

    # Device configuration
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

    # Generate Confusion matrices
    CM(loaded_model, test_dataloader)


    



