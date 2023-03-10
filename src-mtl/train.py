from tqdm.auto import tqdm
import torch
import torch.nn as nn
import numpy as np
from utils import FocalLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, trainloader, optimizer):
    model.train()
    print("Training model...")
    emotion_loss = nn.CrossEntropyLoss()
    real_fake_loss = nn.CrossEntropyLoss()

    # emotion_loss = FocalLoss()
    # real_fake_loss = FocalLoss()

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
        # ------------------------------------------------------------ 
        # Calculate the Loss
        loss_1 = emotion_loss(emotion_output, emotion_label)
        loss_2 = real_fake_loss(real_fake_output, real_fake_label)
        # ------------------------------------------------------------ 
        # IDEA 1: TOTAL LOSS = SUM
        # loss = loss_1 + loss_2
        # total_training_loss += loss
        # ------------------------------------------------------------ 
        # IDEA 2: TOTAL LOSS = PRECISION_1 * LOSS_1 + PRECISION_2 * LOSS_2
        # Calculate precision for emotions
        _, emo_preds = torch.max(emotion_output.data, 1)
        emo_true_positives = (emo_preds == emotion_label).sum().item()
        emo_total_predicted_positives = (emo_preds == emo_preds).sum().item()
        precision_emotions = emo_true_positives / emo_total_predicted_positives
        # Calculate precision for real/fake 
        _, rf_preds = torch.max(real_fake_output.data, 1)
        rf_true_positives = (rf_preds == real_fake_label).sum().item()
        rf_total_predicted_positives = (rf_preds == rf_preds).sum().item()
        precision_real_fake = rf_true_positives / rf_total_predicted_positives
        # Calculation of combined loss
        loss_emotions = precision_emotions * loss_1
        loss_real_fake = precision_real_fake * loss_2
        loss = loss_emotions + loss_real_fake
        total_training_loss += loss
        # ------------------------------------------------------------ 
        # Calculate Accuracy for Emotions
        _, emo_preds = torch.max(emotion_output.data, 1)
        emotion_training_acc += (emo_preds == emotion_label).sum().item()
        # Calculate Accuracy for Real / fake
        _, rf_preds = torch.max(real_fake_output.data, 1)
        real_fake_training_acc += (rf_preds == real_fake_label).sum().item()
        # Calculate Overall Accuracy
        overall_training_acc += (rf_preds == real_fake_label).sum().item()
        overall_training_acc += (emo_preds == emotion_label).sum().item()        
        # ------------------------------------------------------------ 
        # Backpropagation
        loss.backward()
        # Update the weights
        optimizer.step()

    epoch_loss = total_training_loss / counter
    epoch_acc_emotion = 100. * (emotion_training_acc / len(trainloader.dataset))
    epoch_acc_real_fake = 100. * (real_fake_training_acc / len(trainloader.dataset))
    overall_training_acc = 100. * (overall_training_acc / (2*len(trainloader.dataset)))
    return epoch_loss, epoch_acc_emotion, epoch_acc_real_fake, overall_training_acc





def validate(model, testloader):
    model.eval()
    print("Validating model...")
    emotion_loss = nn.CrossEntropyLoss()
    real_fake_loss = nn.CrossEntropyLoss() 
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
            # Calculate Loss
            loss_1 = emotion_loss(emotion_output, emotion_label)
            loss_2 = real_fake_loss(real_fake_output, real_fake_label)
            # ------------------------------------------------------------ 
            # IDEA 1: TOTAL LOSS = SUM
            # loss = loss_1 + loss_2
            # total_validation_loss += loss
            # ------------------------------------------------------------
            # IDEA 2: TOTAL LOSS = PRECISION_1 * LOSS_1 + PRECISION_2 * LOSS_2
            # Calculate precision for emotions
            _, emo_preds = torch.max(emotion_output.data, 1)
            emo_true_positives = (emo_preds == emotion_label).sum().item()
            emo_total_predicted_positives = (emo_preds == emo_preds).sum().item()
            precision_emotions = emo_true_positives / emo_total_predicted_positives
            # Calculate precision for emotions and real/fake 
            _, rf_preds = torch.max(real_fake_output.data, 1)
            rf_true_positives = (rf_preds == real_fake_label).sum().item()
            rf_total_predicted_positives = (rf_preds == rf_preds).sum().item()
            precision_real_fake = rf_true_positives / rf_total_predicted_positives
            # Calculation of combined loss
            loss_emotions = precision_emotions * loss_1
            loss_real_fake = precision_real_fake * loss_2
            loss = loss_emotions + loss_real_fake
            total_validation_loss += loss
            # ------------------------------------------------------------ 
            # Calculate Accuracy for Emotions
            _, emo_preds = torch.max(emotion_output.data, 1)
            emotion_validation_acc += (emo_preds == emotion_label).sum().item()
            # Calculate Accuracy for Real / fake
            _, rf_preds = torch.max(real_fake_output.data, 1)
            real_fake_validation_acc += (rf_preds == real_fake_label).sum().item()
            # Calculate Overall Accuracy
            overall_validation_acc += (rf_preds == real_fake_label).sum().item()
            overall_validation_acc += (emo_preds == emotion_label).sum().item()

    epoch_loss = total_validation_loss / counter
    epoch_acc_emotion = 100. * (emotion_validation_acc / len(testloader.dataset))
    epoch_acc_real_fake = 100. * (real_fake_validation_acc / len(testloader.dataset))
    overall_validation_acc = 100. * (overall_validation_acc / (2*len(testloader.dataset)))
    return epoch_loss, epoch_acc_emotion, epoch_acc_real_fake, overall_validation_acc
