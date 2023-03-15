import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import VideoDataset
from tqdm.auto import tqdm

from dataset import VideoDataset, get_data_loaders
from utils import save_plots

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader, val_dataloader = get_data_loaders('data_sequences/train_seq',
                                                    'data_sequences/val_seq',
                                                    seq_length=20,
                                                    batch_size=32,
                                                    num_workers=4)

class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(EmotionRecognitionModel, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.bn3 = nn.BatchNorm3d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(125440, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x



class EmotionRecognitionModel_Bigger(nn.Module):
    def __init__(self, num_classes):
        super(EmotionRecognitionModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.bn3 = nn.BatchNorm3d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(250880, 1024)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x

    

# Get the model
model = EmotionRecognitionModel(num_classes=12).to(device)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()

###### Define optimizer
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)

optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)



# Train the model
num_epochs = 10
train_loss = []
train_acc = []
val_loss = []
val_acc = []
for epoch in range(num_epochs):
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    running_loss = 0.0

    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        inputs, labels = data
        inputs = inputs.permute(0, 4, 1, 2, 3)

        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        train_running_correct += (predicted == labels).sum().item()

        # Calculate Loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        counter += 1

    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(train_dataloader.dataset))
    train_loss.append(epoch_loss) # for plotting
    train_acc.append(epoch_acc) # for plotting
    print('Epoch [{}/{}], Training Loss: {:.4f}, Training Accuracy: {:.4f}%' % (epoch+1, num_epochs, epoch_loss, epoch_acc))
    

    # Evaluate the model on the validation set
    correct = 0
    total = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            inputs, labels = data
            inputs = inputs.permute(0, 4, 1, 2, 3)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * (correct / len(val_dataloader.dataset))
    
    val_loss.append(epoch_loss) # for plotting
    val_acc.append(epoch_acc) # for plotting
    print('Epoch [{}/{}], Validation Loss: {:.4f}, Validation Accuracy: {:.4f}%' % (epoch+1, num_epochs, epoch_loss, epoch_acc))
    
    # print('Epoch [%d], Validation Loss: %.4f, Validation Accuracy: %.4f' %
    #       (epoch+1, running_loss/len(train_dataloader), correct/total))


# Plot the loss and accuracy curves

save_plots(train_acc, val_acc, train_loss, val_loss)