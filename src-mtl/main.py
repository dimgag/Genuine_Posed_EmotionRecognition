import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import os
from PIL import Image

from dataset import SASEFE_MTL, SASEFE_MTL_TEST
from utils import save_model, get_model_params, save_plots, freeze_baseline
from models import HydraNet, ChimeraNet, ChimeraNetV2
from train import train, validate


def main():
    # Data directory
    train_dir = "data_mtl/train"
    test_dir = "data_mtl/test"
    
    # With Facial Landmarks
    # train_dir = "data_mtl_withfacialandmarks/train"
    # test_dir = "data_mtl_withfacialandmarks/test"
    
    train_image_paths = os.listdir(train_dir)
    test_image_paths = os.listdir(test_dir)

    # Get the dataset class
    train_dataset = SASEFE_MTL(train_image_paths)
    test_dataset = SASEFE_MTL_TEST(test_image_paths)

    # Get the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    print("Number of training images: ", len(train_dataset))
    print("Number of test images: ", len(test_dataset))
    # Print dataloaders 
    print("Train dataloader: ", len(train_dataloader))
    print("Test dataloader: ", len(test_dataloader))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    

    
    # print("Device:", device)
    
    # Define the model
    # net = HydraNet()
    # model = HydraNet().to(device)
    # model = ChimeraNet().to(device)
    model = ChimeraNetV2().to(device)

    get_model_params(model)

    # Fine Tuning the model
    # model = freeze_baseline(model) # Freeze the baseline model and train only the new layers
    

    # Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # Train & Validate the model 
    train_loss = []
    valid_loss = []
    train_loss1 = []
    valid_loss1 = []
    train_emo_acc = []
    valid_emo_acc = []
    train_real_fake_acc = []
    valid_real_fake_acc = []
    total_train_acc = []
    total_valid_acc = []

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Set the number of epochs here.
    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        # Train the model.
        train_epoch_loss, train_emo_epoch_acc, train_real_fake_epoch_acc, overall_training_acc  = train(model, train_dataloader, optimizer)
        # Validation of the model.
        valid_epoch_loss, valid_emo_epoch_acc, valid_real_fake_epoch_acc, overall_validation_acc = validate(model, test_dataloader)
        
        # Save the loss and accuracy for the epoch. 
        train_loss1.append(train_epoch_loss)
        valid_loss1.append(valid_epoch_loss)
        train_loss.append(train_loss1[0].tolist())
        valid_loss.append(valid_loss1[0].tolist())
        
        train_loss1 = []
        valid_loss1 = []
        
        train_emo_acc.append(train_emo_epoch_acc)
        valid_emo_acc.append(valid_emo_epoch_acc)

        train_real_fake_acc.append(train_real_fake_epoch_acc) # For plotting
        valid_real_fake_acc.append(valid_real_fake_epoch_acc) # For plotting

        total_train_acc.append(overall_training_acc) # For plotting
        total_valid_acc.append(overall_validation_acc) # For plotting
        
        # Update the learning rate. -if using scheduler- If not, comment the next line.
        scheduler.step(valid_epoch_loss)

        # Print the loss and accuracy for the epoch.
        print(f"Training loss: {train_epoch_loss:.3f}, Emotion training acc: {train_emo_epoch_acc:.3f}, Real/Fake training acc: {train_real_fake_epoch_acc:.3f}, Overall training acc: {overall_training_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, Emotion validation acc: {valid_emo_epoch_acc:.3f}, Real/Fake validation acc: {valid_real_fake_epoch_acc:.3f}, Overall Validation acc: {overall_validation_acc:.3f}")
        print('-'*50)

    print('Finished Training') 
    save_model(epochs, model, optimizer)
    save_plots(train_emo_acc, valid_emo_acc, train_real_fake_acc, valid_real_fake_acc, train_loss, valid_loss, total_train_acc, total_valid_acc)
    


if __name__ == "__main__":
    main()
