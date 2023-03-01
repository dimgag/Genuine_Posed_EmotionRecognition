import torch
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
matplotlib.style.use('ggplot')

def save_model(epochs, model, optimizer, criterion):
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, 'model.pth')


def get_model_params(model):
  """Get model parameters"""
  total_params = sum(p.numel() for p in model.parameters())
  print(f"Total parameters: {total_params:,}")
  total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"Trainable parameters: {total_trainable_params:,}")

def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"accuracy.png")
    
    # Loss plots.
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




def ConfusionMatrix(net, test_loader, dataset_classes):
    '''Function to plot confusion matrix'''
    y_pred = []
    y_true = []
    # iterate over test data
    for inputs, labels in test_loader:
            output = net(inputs.cuda()) # Feed Network - Remove .cuda() for CPU usage

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
            
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth

    # constant for classes
    classes = dataset_classes

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sns.heatmap(df_cm, annot=True)
    plt.savefig('output.png')



def ConfusionMatrix_MT(model, test_loader, real_fake_classes, emotion_classes):
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


