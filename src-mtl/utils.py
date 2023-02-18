import torch
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

matplotlib.style.use('ggplot')


def save_model(epochs, model, optimizer):
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, 'model.pth')



def get_model_params(model):
  """Get model parameters"""
  total_params = sum(p.numel() for p in model.parameters())
  print(f"Total parameters: {total_params:,}")
  total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"Trainable parameters: {total_trainable_params:,}")

def freeze_baseline(model):
	print("-"*50)
	print("Model parameters before freezing the model:", get_model_params(model))
    
	for param in model.parameters():
		param.requires_grad = False
        
	for param in model.net.fc1.parameters():
		param.requires_grad = True
        
	for param in model.net.fc2.parameters():
		param.requires_grad = True

	print("-"*50)
	print("\nModel parameters after freezing the baseline:", get_model_params(model))
    
	return model


def save_plots(train_emo_acc, valid_emo_acc, train_real_fake_acc, valid_real_fake_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plot for Emotions
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_emo_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_emo_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy plot for Emotions')
    plt.legend()
    plt.savefig(f"accuracy_emo.png")
    
    # Accuracy plot for real/fake
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_real_fake_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_real_fake_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy plot for Real/Fake')
    plt.legend()
    plt.savefig(f"accuracy_rf.png")

    # Loss plots.
    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)
    
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"loss.png")


def ConfusionMatrix_MT(model, test_loader, real_fake_classes, emotion_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_pred_rf = []
    y_true_rf = []

    y_pred_emo = []
    y_true_emo = []

    for inputs, labels in enumerate(test_loader):
        inputs = labels["image"].to(device)
        real_fake_labels = labels["real_fake"].data.cpu().numpy()
        emotion_labels = labels["emotion"].data.cpu().numpy()
        real_fake_output, emotion_output = model(inputs)
        real_fake_output = (torch.max(torch.exp(real_fake_output), 1)[1]).data.cpu().numpy()
        emotion_output = (torch.max(torch.exp(emotion_output), 1)[1]).data.cpu().numpy()
        y_pred_rf.extend(real_fake_output)
        y_pred_emo.extend(emotion_output) 
        y_true_rf.extend(real_fake_labels)
        y_true_emo.extend(emotion_labels)

    rf_classes = real_fake_classes
    emo_classes = emotion_classes

    # Build Confusion Matrix
    cf_matrix_rf = confusion_matrix(y_true_rf, y_pred_rf)
    cf_matrix_emo = confusion_matrix(y_true_emo, y_pred_emo)

    df_cm_rf = pd.DataFrame(cf_matrix_rf/np.sum(cf_matrix_rf) *100, index = [i for i in rf_classes], columns = [i for i in rf_classes])
    df_cm_emo = pd.DataFrame(cf_matrix_emo/np.sum(cf_matrix_emo) *100, index = [i for i in emo_classes], columns = [i for i in emo_classes])

    plt.figure(figsize=(12,7))
    sns.heatmap(df_cm_rf, annot=True)
    plt.savefig('cm_real_fake.png')

    plt.figure(figsize=(12,7))
    sns.heatmap(df_cm_emo, annot=True)
    plt.savefig('cm_emotions.png')
    



