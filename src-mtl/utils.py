import torch
import seaborn as sn
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



# def ConfusionMatrix(net, test_loader, dataset_classes):
#     y_pred = []
#     y_true = []
#     # iterate over test data
#     for inputs, labels in test_loader:
#             output = net(inputs.cuda()) # Feed Network - Remove .cuda() for CPU usage

#             output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
#             y_pred.extend(output) # Save Prediction
            
#             labels = labels.data.cpu().numpy()
#             y_true.extend(labels) # Save Truth

#     # constant for classes
#     classes = dataset_classes

#     # Build confusion matrix
#     cf_matrix = confusion_matrix(y_true, y_pred)
#     df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
#                         columns = [i for i in classes])
#     plt.figure(figsize = (12,7))
#     sn.heatmap(df_cm, annot=True)
#     plt.savefig('output.png')



def cm_emotions(net, test_loader, emotion_labels):
    y_pred = []
    y_true = []
    # iterate over test data
    for inputs, labels in test_loader:
            output = net(inputs.cuda()) # Feed Network - Remove .cuda() for CPU usage

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
            
            labels = labels["emotion"].cpu().numpy()
            # labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth

    # constant for classes
    classes = emotion_labels

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output.png')