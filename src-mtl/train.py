from tqdm.auto import tqdm
import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, trainloader, optimizer):
    model.train()
    print("Training model...")
    
    # Define the loss functions.
    emotion_loss = nn.CrossEntropyLoss()  # Includes Softmax
    # real_fake_loss = nn.BCELoss() # Doesn't include Softmax
    real_fake_loss = nn.CrossEntropyLoss() 
    
    # Define the sigmoid function
    # Sig = nn.Sigmoid()
    total_training_loss = 0.0
    emotion_training_acc = 0
    real_fake_training_acc = 0
    overall_training_acc = 0
    counter = 0

    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        inputs = data["image"].to(device)
        real_fake_label = data["real_fake"].to(device) 
        emotion_label = data["emotion"].to(device)

        # Clear the gradients
        optimizer.zero_grad()
        # Forward pass
        real_fake_output, emotion_output = model(inputs)
        # Calculate the Loss
        loss_1 = emotion_loss(emotion_output, emotion_label)

        # loss_2 = real_fake_loss(Sig(real_fake_output), real_fake_label.unsqueeze(1).float()) # This is for BCELoss (Also in valide)
        # Try some thing here. 
        loss_2 = real_fake_loss(real_fake_output, real_fake_label)
        loss = loss_1 + loss_2
        total_training_loss += loss
        

        # ------------------------------------------------------------
        # Idea: to calculate the loss based on the precision scores.
        # Calculate precision for emotions and real/fake 
        '''
        precisiotn_emotions = emotion_output.data.max(1, keepdim=True)[1] == emotion_label.data.max(1, keepdim=True)[1]
        precisiotn_real_fake = real_fake_output.data.max(1, keepdim=True)[1] == real_fake_label.data.max(1, keepdim=True)[1]
        
        # Calculate the Loss
        loss_emotions = precisiotn_emotions * loss_1
        loss_real_fake = precisiotn_real_fake * loss_2

        loss = loss_emotions + loss_real_fake
        total_training_loss += loss
        '''
        # ------------------------------------------------------------ 

        # Calculate Accuracy for Emotions
        _, emo_preds = torch.max(emotion_output.data, 1)
        emotion_training_acc += (emo_preds == emotion_label).sum().item()
        
        # Calculate Accuracy for Real / fake
        _, rf_preds = torch.max(real_fake_output.data, 1)        
        real_fake_training_acc += (rf_preds == real_fake_label).sum().item()
        
        # Calculate Overall Accuracy
        # _, rf_preds = torch.max(real_fake_output.data, 1)
        # _, emo_preds = torch.max(emotion_output.data, 1)        
        overall_training_acc = (rf_preds == real_fake_label).sum().item()
        overall_training_acc += (emo_preds == emotion_label).sum().item()

        # Backpropagation
        loss.backward()
        # Update the weights
        optimizer.step()

    epoch_loss = total_training_loss / counter
    epoch_acc_emotion = 100. * (emotion_training_acc / len(trainloader.dataset))
    epoch_acc_real_fake = 100. * (real_fake_training_acc / len(trainloader.dataset))
    overall_training_acc = 100. * (overall_training_acc / len(trainloader.dataset))


    return epoch_loss, epoch_acc_emotion, epoch_acc_real_fake, overall_training_acc





def validate(model, testloader):
    model.eval()
    print("Validating model...")
    
    # Define the loss functions.
    emotion_loss = nn.CrossEntropyLoss() # Includes Softmax
    # real_fake_loss = nn.BCELoss() # Doesn't include Softmax
    real_fake_loss = nn.CrossEntropyLoss() 

    # Sig = nn.Sigmoid()
    total_validation_loss = 0.0
    emotion_validation_acc = 0
    real_fake_validation_acc = 0 
    overall_validation_acc = 0
    counter = 0
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter +=1
            inputs = data["image"].to(device)

            real_fake_label = data["real_fake"].to(device) 
            emotion_label = data["emotion"].to(device)

            # Forward pass
            real_fake_output, emotion_output = model(inputs)

            # ------------------------------------------------------------
            # Calculate the Loss and Sum them together.
            loss_1 = emotion_loss(emotion_output, emotion_label)
            
            # loss_2 = real_fake_loss(Sig(real_fake_output), real_fake_label.unsqueeze(1).float())
            loss_2 = real_fake_loss(real_fake_output, real_fake_label)
            
            loss = loss_1 + loss_2
            total_validation_loss += loss

            # ------------------------------------------------------------
            # Idea: to calculate the loss based on the precision scores.
            # Calculate precision for emotions and real/fake 
            '''
            precisiotn_emotions = emotion_output.data.max(1, keepdim=True)[1] == emotion_label.data.max(1, keepdim=True)[1]
            precisiotn_real_fake = real_fake_output.data.max(1, keepdim=True)[1] == real_fake_label.data.max(1, keepdim=True)[1]
            
            # Calculate the Loss
            loss_emotions = precisiotn_emotions * loss_1
            loss_real_fake = precisiotn_real_fake * loss_2

            loss = loss_emotions + loss_real_fake
            total_validation_loss += loss
            '''
            # ------------------------------------------------------------ 


            # Calculate Accuracy for emotions
            _, emo_preds = torch.max(emotion_output.data, 1)
            emotion_validation_acc += (emo_preds == emotion_label).sum().item()
            # Calculate Accuracy for Real / fake
            _, rf_preds = torch.max(real_fake_output.data, 1)
            real_fake_validation_acc += (rf_preds == real_fake_label).sum().item()
            
            
            # Calculate Overall Accuracy
            _, rf_preds = torch.max(real_fake_output.data, 1) # remove this?
            _, emo_preds = torch.max(emotion_output.data, 1) # Remove this?

            overall_validation_acc = (rf_preds == real_fake_label).sum().item()
            overall_validation_acc += (emo_preds == emotion_label).sum().item()


        epoch_loss = total_validation_loss / counter
        epoch_acc_emotion = 100. * (emotion_validation_acc / len(testloader.dataset))
        epoch_acc_real_fake = 100. * (real_fake_validation_acc / len(testloader.dataset))
        overall_validation_acc = 100. * (overall_validation_acc / len(testloader.dataset))
    return epoch_loss, epoch_acc_emotion, epoch_acc_real_fake, overall_validation_acc
