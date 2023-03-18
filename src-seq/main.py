import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from dataset import VideoDataset
from tqdm.auto import tqdm

from dataset import VideoDataset, get_data_loaders
from models import MyNetwork, EmotionRecognitionModel, EmotionRecognitionModel2, EmotionRecognitionModel_Bigger
from utils import save_plots, save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

train_dataloader, val_dataloader = get_data_loaders('data_sequences/train_seq',
                                                    'data_sequences/val_seq',
                                                    seq_length=20,
                                                    batch_size=24,
                                                    num_workers=4)


## Define the Model

# model = MyNetwork(num_classes=12).to(device)


model = EmotionRecognitionModel(num_classes=12).to(device)


# model = EmotionRecognitionModel2(num_classes=12).to(device)

# model = EmotionRecognitionModel_Bigger(num_classes=12).to(device)

## Loss Function
criterion = nn.CrossEntropyLoss()



##Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# optimizer = optim.Adam(model.parameters(), lr=0.01)#, weight_decay=0.001)

# optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)

# optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

## LR-Scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)




## Train the model
num_epochs = 500
train_loss = []
train_acc = []
val_loss = []
val_acc = []
for epoch in range(num_epochs):
    print("\nTraining...")
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
        counter += 1

    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(train_dataloader.dataset))
    train_loss.append(epoch_loss) # for plotting
    train_acc.append(epoch_acc) # for plotting
    print('Epoch [%d], Training Loss: %.4f, Training Accuracy: %.4f' % (epoch+1, epoch_loss, epoch_acc))


    # Evaluate the model on the validation set
    correct = 0
    total = 0
    running_loss = 0.0
    print("Validating...")
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            inputs, labels = data
            inputs = inputs.permute(0, 4, 1, 2, 3)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            correct += (predicted == labels).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = 100. * (correct / len(val_dataloader.dataset))
    val_loss.append(epoch_loss) # for plotting
    val_acc.append(epoch_acc) # for plotting
    print('Epoch [%d], Validation Loss: %.4f, Validation Accuracy: %.4f' % (epoch+1, epoch_loss, epoch_acc))
    scheduler.step(epoch_loss)

# Plot the loss and accuracy curves
save_plots(train_acc, val_acc, train_loss, val_loss)
# save the model
save_model(num_epochs, model, optimizer, criterion)



######################################################################
## Evaluate the model on validation set.
path = 'model.pth'

loaded_model = model

loaded_checkpoint = torch.load(path)
loaded_model.load_state_dict(loaded_checkpoint['model_state_dict'])
optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
epoch = loaded_checkpoint['epoch']

loaded_model.eval()
correct = 0
total = 0
print("Evaluate the model.")
with torch.no_grad():
    running_loss = 0.0
    for i, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        inputs, labels = data
        inputs = inputs.permute(0, 4, 1, 2, 3)
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * (correct / len(val_dataloader.dataset))
    print(f"Test loss: {epoch_loss:.3f}, Test acc: {epoch_acc:.3f}")



