# Main file for the project
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Import modules
from utils import save_model, save_plots
from data import get_datasets, get_data_loaders
from train import train, validate



if __name__ == '__main__':
    # 1. Load the data 
    # Local Paths
    train_dir = "data/train"
    test_dir = "data/test"

    # train_images = os.listdir(train_dir)
    # test_images = os.listdir(test_dir)

    dataset_train, dataset_test, dataset_classes = get_datasets()
    train_loader, test_loader = get_data_loaders(dataset_train, dataset_test)


    # 2. Define the model
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 53 * 53, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 12)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            # x = x.view(-1, 16 * 53 * 53)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()

    # 3. Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
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


    # Save the Model and Accuracy&Loss plots.
    save_model(epochs, net, optimizer, criterion)
    
    save_plots(train_acc, valid_acc, train_loss, valid_loss)
    print('TRAINING COMPLETE')