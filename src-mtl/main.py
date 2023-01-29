# HydraNet - Multi Task approach for Real/Fake and Emotion Classification
# The dataset is a set of images, ana the names of the images give the labels.
# For example the image "real_happy_1.jpg" is a real image of a happy person.

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import os
from PIL import Image

# Import modues
from dataset import SASEFE_MTL, SASEFE_MTL_TEST
# from utils import *
from utils import save_model, get_model_params, save_plots, cm_emotions, freeze_baseline

from models import HydraNet, ChimeraNet

from train import train, validate



def main():
    # Data directory
    train_dir = "data_mtl/train"
    test_dir = "data_mtl/test"
    train_image_paths = os.listdir(train_dir)
    test_image_paths = os.listdir(test_dir)


    # Get the dataset class
    train_dataset = SASEFE_MTL(train_image_paths)
    test_dataset = SASEFE_MTL_TEST(test_image_paths)

    # Get the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    print("Number of training images: ", len(train_dataset))
    print("Number of test images: ", len(test_dataset))
    # Print dataloaders 
    print("Train dataloader: ", len(train_dataloader))
    print("Test dataloader: ", len(test_dataloader))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Get the model
    # net = HydraNet()
    # model = HydraNet(net).to(device)

    model = ChimeraNet().to(device)

    # get model parameters
    # get_model_params(model)
    model = freeze_baseline(model)
    



    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.09)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # Train & Val the model 
    train_loss = []
    valid_loss = []
    
    train_loss1 = []
    valid_loss1 = []

    train_emo_acc = []
    valid_emo_acc = []

    train_real_fake_acc = []
    valid_real_fake_acc = []

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        # Train the model.
        train_epoch_loss, train_emo_epoch_acc, train_real_fake_epoch_acc  = train(model, train_dataloader, optimizer)
        # Validation of the model.
        valid_epoch_loss, valid_emo_epoch_acc, valid_real_fake_epoch_acc = validate(model, test_dataloader)
        
        # Save the loss and accuracy for the epoch. 
        train_loss1.append(train_epoch_loss)
        valid_loss1.append(valid_epoch_loss)
        

        
        train_loss.append(train_loss1[0].tolist())
        valid_loss.append(valid_loss1[0].tolist())
        
        
        train_loss1 = []
        valid_loss1 = []
        
        #######################################

        train_emo_acc.append(train_emo_epoch_acc)
        valid_emo_acc.append(valid_emo_epoch_acc)

        train_real_fake_acc.append(train_real_fake_epoch_acc)
        valid_real_fake_acc.append(valid_real_fake_epoch_acc)
        
        # Update the learning rate. -if using scheduler-
        scheduler.step(valid_epoch_loss)

        # Print the loss and accuracy for the epoch.
        print(f"Training loss: {train_epoch_loss:.3f}, Emotion training acc: {train_emo_epoch_acc:.3f}, Real/Fake training acc: {train_real_fake_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, Emotion validation acc: {valid_emo_epoch_acc:.3f}, Real/Fake validation acc: {valid_real_fake_epoch_acc:.3f}")
        print('-'*50)

    print('Finished Training') 

    # In the end of the training I have the following lists:
    # train_loss, valid_loss, train_emo_acc, valid_emo_acc, train_real_fake_acc, valid_real_fake_acc
    # One plot for the losses and one plot for the accuracies.
    # 'fake_contempt_H2N2C.MP4Anton274.jpg'

    save_model(epochs, model, optimizer)
    save_plots(train_emo_acc, valid_emo_acc, train_real_fake_acc, valid_real_fake_acc, train_loss, valid_loss)

    # cm_emotions(model, test_dataloader, 6)




if __name__ == "__main__":
    main()
