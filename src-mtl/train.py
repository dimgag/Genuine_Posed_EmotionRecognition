from tqdm.auto import tqdm
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, trainloader, optimizer):
    model.train()
    print("Training model...")
    
    Sig = nn.Sigmoid()
    total_training_loss = 0.0
    emotion_training_acc = 0
    real_fake_training_acc = 0 
    counter = 0

    for i, data in tqdm(enumerate(trainloader)):
        inputs = data["image"].to(device)

        real_fake_label = data["real_fake"].to(device)
        emotion_label = data["emotion"].to(device)
        optimizer.zero_grad()

        # Forward pass
        real_fake_output, emotion_output = model(inputs)

        # Calculate the Loss
        loss_1 = emotion_loss(emotion_output, emotion_label)
        loss_2 = real_fake_loss(Sig(real_fake_output), real_fake_label.unsqueeze(1).float())
        loss = loss_1 + loss_2
        total_training_loss += loss
        
        # Calculate Accuracy
        _, preds = torch.max(emotion_output.data, 1)
        
        _, preds = torch.max(real_fake_output.data, 1)

        emotion_training_acc += (preds == emotion_label).sum().item()
        
        real_fake_training_acc += (preds == real_fake_label).sum().item()

        loss.backward()
        optimizer.step()

    epoch_loss = total_training_loss / counter
    epoch_acc_emotion = 100. * (emotion_training_acc / len(trainloader.dataset))
    epoch_acc_real_fake = 100. * (real_fake_training_acc / len(trainloader.dataset))
    
    return epoch_loss, epoch_acc_emotion, epoch_acc_real_fake





def validate(model, testloader):
    model.eval()
    print("Validating model...")
    
    Sig = nn.Sigmoid()
    total_validation_loss = 0.0
    emotion_validation_acc = 0
    real_fake_validation_acc = 0 
    counter = 0

    for i, data in tqdm(enumerate(testloader)):
        inputs = data["image"].to(device)

        real_fake_label = data["real_fake"].to(device)
        emotion_label = data["emotion"].to(device)

        # Forward pass
        real_fake_output, emotion_output = model(inputs)

        # Calculate the Loss
        loss_1 = emotion_loss(emotion_output, emotion_label)
        loss_2 = real_fake_loss(Sig(real_fake_output), real_fake_label.unsqueeze(1).float())
        loss = loss_1 + loss_2
        total_validation_loss += loss
        
        # Calculate Accuracy
        _, preds = torch.max(emotion_output.data, 1)
        
        _, preds = torch.max(real_fake_output.data, 1)

        emotion_validation_acc += (preds == emotion_label).sum().item()
        
        real_fake_validation_acc += (preds == real_fake_label).sum().item()

    epoch_loss = total_validation_loss / counter
    epoch_acc_emotion = 100. * (emotion_validation_acc / len(testloader.dataset))
    epoch_acc_real_fake = 100. * (real_fake_validation_acc / len(testloader.dataset))
    
    return epoch_loss, epoch_acc_emotion, epoch_acc_real_fake



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Add to main.py
# 
# Define the Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.09)

# Define the Losses
# L_1: the emotion Loss, is a multi-class classification loss. In our case, it’s Cross-Entropy!
# L_2: the Real_Fake Loss, is a Binary Classification loss. In our case, Binary Cross-Entropy.
emotion_loss = nn.CrossEntropyLoss() # Includes Softmax
real_fake_loss = nn.BCELoss() # Doesn't include Softmax
# Maybe you don't have to put those in the main.py, but I did it for clarity....


