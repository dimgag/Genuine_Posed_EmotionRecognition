import torch
import seaborn as sns
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


def CM(model, test_loader, real_fake_classes, emotion_classes):
    y_pred_rf = []
    y_true_rf = []

    y_pred_emo = []
    y_true_emo = []

    for inputs, labels in enumerate(test_loader):
        inputs = labels["image"]

        real_fake_labels = labels["real_fake"].data.cpu().numpy()
        emotion_labels = labels["emotion"].data.cpu().numpy()

        real_fake_output, emotion_output = model(inputs)

        real_fake_output = (torch.max(torch.exp(real_fake_output), 1)[1]).data.cpu().numpy()
        emotion_output = (torch.max(torch.exp(emotion_output), 1)[1]).data.cpu().numpy()

        y_pred_rf.extend(real_fake_output)
        y_pred_emo.extend(emotion_output) 

        y_true_rf.extend(real_fake_labels)
        y_true_emo.extend(emotion_labels)


        # print(real_fake_labels)

    rf_classes = real_fake_classes
    emo_classes = emotion_classes


    # Build Confusion Matrix
    cf_matrix_rf = confusion_matrix(y_true_rf, y_pred_rf)
    cf_matrix_emo = confusion_matrix(y_true_emo, y_pred_emo)

    df_cm_rf = pd.DataFrame(cf_matrix_rf/np.sum(cf_matrix_rf) *10, index = [i for i in rf_classes], columns = [i for i in rf_classes])
    df_cm_emo = pd.DataFrame(cf_matrix_emo/np.sum(cf_matrix_emo) *10, index = [i for i in emo_classes], columns = [i for i in emo_classes])

    plt.figure(figsize=(12,7))
    sns.heatmap(df_cm_rf, annot=True)
    plt.savefig('cm_real_fake.png')

    plt.figure(figsize=(12,7))
    sns.heatmap(df_cm_emo, annot=True)
    plt.savefig('cm_emotions.png')


    # # 
    # counter = 0
    
    # for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
    #     counter += 1
    #     inputs = data["image"].to(device)
    #     real_fake_label = data["real_fake"].to(device) 
    #     emotion_label = data["emotion"].to(device)

    #     real_fake_output, emotion_output = model(inputs)
        
        
    #     # So I have emotion_output = prediction and emotion_label = GT
    #     # generate Confussion matrix for real_fake: 
    
    # real_fake_output = real_fake_output.cpu()
    # real_fake_output = real_fake_output.detach().numpy()
    # real_fake_output = np.round(real_fake_output)
    # # emotion_output = np.argmax(emotion_output, axis=0)
    
    
    # real_fake_cm = confusion_matrix(real_fake_classes, real_fake_output)
    # print(real_fake_cm)




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
    # print(test_dataset.__getitem__(0))

    # # Get the dataloaders
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    # Get the classes of the dataset to generate the confusion matrix
    real_fake_classes = test_dataset.real_fakes
    real_fake_classes = np.unique(real_fake_classes)
    convert_dict = {0: 'fake', 1: 'real'}
    real_fake_classes = [convert_dict.get(i, i) for i in real_fake_classes]
    # print(real_fake_classes)

    emotion_classes = test_dataset.emotions
    emotion_classes = np.unique(emotion_classes)
    convert_dict = {0: 'happy', 1: 'sad', 2: 'surprise', 3: 'disgust', 4: 'contempt', 5: 'angry'}
    emotion_classes = [convert_dict.get(i, i) for i in emotion_classes]
    # print(emotion_classes)

    CM(loaded_model, test_dataloader, real_fake_classes, emotion_classes)
