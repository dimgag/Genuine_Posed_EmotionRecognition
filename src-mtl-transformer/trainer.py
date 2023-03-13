import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, val_loader, optimizer, epochs):
    print("Training model...")
    for epoch in range(epochs):
        model.train()
        emotion_loss = nn.CrossEntropyLoss()
        real_fake_loss = nn.CrossEntropyLoss()
        total_training_loss = 0.0
        emotion_training_acc = 0
        real_fake_training_acc = 0
        overall_training_acc = 0
        counter = 0
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()
            # how to make the inputs and labels into tensors?
            inputs = data["frames"].to(device)
            # print the shape of inputs 
            print("Shape of the inputs: ", inputs.shape)
            
            print(inputs.shape)

            real_fake_label = data["rf_label"].to(device)
            emotion_label = data["emo_label"].to(device)
            real_fake_output, emotion_output = model(inputs)
            # ------------------------------------------------------------
            # Calculate the Losses
            loss_1 = emotion_loss(emotion_output, emotion_label)
            loss_2 = real_fake_loss(real_fake_output, real_fake_label)
            # ------------------------------------------------------------
            # Calculate Precision for emotions
            _, emo_preds = torch.max(emotion_output.data, 1)
            emo_true_positives = (emo_preds == emotion_label).sum().item()
            emo_total_predicted_positives = (emo_preds == emo_preds).sum().item()
            precision_emotions = emo_true_positives / emo_total_predicted_positives
            # Calculate Precision for real/fake
            _, rf_preds = torch.max(real_fake_output.data, 1)
            rf_true_positives = (rf_preds == real_fake_label).sum().item()
            rf_total_predicted_positives = (rf_preds == rf_preds).sum().item()
            precision_real_fake = rf_true_positives / rf_total_predicted_positives
            # Calculate the combined loss
            loss_emotions = precision_emotions * loss_1
            loss_real_fake = precision_emotions * loss_2
            loss = loss_emotions + loss_real_fake
            total_training_loss += loss
            # ------------------------------------------------------------
            # Calculate Accuracy for Emotions
            _, emo_preds = torch.max(emotion_output.data, 1)
            emotion_training_acc += (emo_preds == emotion_label).sum().item()
            # Calculate Accuracy for Real/Fake
            _, rf_preds = torch.max(real_fake_output.data, 1)
            real_fake_training_acc += (rf_preds == real_fake_label).sum().item()
            # Calculate overall accuracy
            overall_training_acc += (rf_preds == real_fake_label).sum().item()
            overall_training_acc += (emo_preds == emotion_label).sum().item()
            # Backpropagation
            loss.backward()
            # Update the weights
            optimizer.step()
        epoch_loss = total_training_loss / counter 
        epoch_acc_emotion = 100. * (emotion_training_acc / len(train_loader.dataset))
        epoch_acc_real_fake = 100. * (real_fake_training_acc / len(train_loader.dataset))
        overall_training_acc = 100. * (overall_training_acc / (2*len(train_loader.dataset)))
        return epoch_loss, epoch_acc_emotion, epoch_acc_real_fake, overall_training_acc
