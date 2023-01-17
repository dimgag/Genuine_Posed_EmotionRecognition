# Main file for the project
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from utils import save_model, save_plots, get_model_params, ConfusionMatrix
from data import get_datasets, get_data_loaders
from train import train, validate
from fine_tune import freeze_model, get_model, get_model_params, add_classification_head
from models.model1 import Net
from models.vggface import VGGFace, VGGFace2
from models.facenet import FaceNet, FaceNet_withClassifier


import loss_functions


if __name__ == '__main__':
    # Device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1. Load the data 
    # Local Paths
    train_dir = "data/train"
    test_dir = "data/test"

    # train_images = os.listdir(train_dir)
    # test_images = os.listdir(test_dir)

    dataset_train, dataset_test, dataset_classes = get_datasets()
    train_loader, test_loader = get_data_loaders(dataset_train, dataset_test)
    

    # 2. Define the model to train in gpu
    # net = Net().to(device)
    # net = VGGFace().to(device)
    # net = VGGFace2().to(device)
    # net = FaceNet().to(device)
    net = FaceNet_withClassifier().to(device)

    # Fine Tuning the model:
    net = freeze_model(net)
    # net = add_classification_head(net, device)


    # Print model parameters
    # get_model_params(net)

    # 3. Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # criterion = loss_functions.TripletMarginLoss()
    # criterion = loss_functions.MultiFocalLoss()

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 4. Train the model
    # Lists to keep track of the loss and accuracy.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


    epochs = 30

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        # Train the model.
        train_epoch_loss, train_epoch_acc = train(net, train_loader, optimizer, criterion)
        # Validation of the model.
        valid_epoch_loss, valid_epoch_acc = validate(net, test_loader, criterion, dataset_classes)
        # Save the loss and accuracy for the epoch.
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)

        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)

    print('Finished Training')



    # Save the Model, Accuracy & Loss plots and Confusion Matrix
    save_model(epochs, net, optimizer, criterion)    
    save_plots(train_acc, valid_acc, train_loss, valid_loss)
    ConfusionMatrix(net, test_loader, dataset_classes)
