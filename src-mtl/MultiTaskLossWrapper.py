import torch
import torch.nn as nn

# idea1: Combine the two losses into one loss function based on the precision scores. 

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num, model):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = model
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, input, targets):
        
        outputs = self.model(input)

        precision1 = torch.exp(-self.log_vars[0])
        loss = torch.sum(precision1 * (targets[0] - outputs[0]) ** 2. + self.log_vars[0], -1)


        precision2 = torch.exp(-self.log_vars[1])
        loss += torch.sum(precision2 * (targets[1] - outputs[1]) ** 2. + self.log_vars[1], -1)

        loss = torch.mean(loss)


        return loss, self.log_vars.data.tolist()

        ########################################################################
'''        # Calculate the losses, as you did in the original code
        # Edit this code ? to make it work somehow... after the calculation of the los

        crossEntropy = nn.CrossEntropyLoss()

        loss_real_fake = crossEntropy(real_fake_prediction, real_fake_label) # To adjust. I have to add this to the training loop. train.py (and validation loop)
        
        loss_emotions = crossEntropy(emotion_prediction, emotio_label) # To adjust.

        precision_real_fake = torch.exp(-self.log_vars[0])
        loss_real_fake = precision_real_fake * loss_real_fake + self.log_vars[0]

        precision_emotions = torch.exp(-self.log_vars[1])
        loss_emotions = precision_emotions * loss_emotions + self.log_vars[1]
		
        return loss_real_fake + loss_emotions
'''




# Idea2: combine the two losses into one loss function based on the F1-scores

# Implement F1-score function
import numpy as np 
def f1_score(y_true, y_pred):
    """
    Function to calculate the F1 score.
    """
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

# Implement accuracy function
def f1_loss(outputs, labels, f1_fn):
    # Get the outputs for each task
    task1_outputs = outputs[0]
    task2_outputs = outputs[1]

    # Get the labels for each task
    task1_labels = labels[0]
    task2_labels = labels[1]

    # Calculate the F1 scores for each task
    task1_f1 = f1_fn(task1_outputs, task1_labels)
    task2_f1 = f1_fn(task2_outputs, task2_labels)

    # Calculate the total loss as a weighted sum of the F1 scores
    total_loss = 1 - (task1_f1 + task2_f1) / 2

    return total_loss

# loss = f1_loss(outputs, labels, f1_score)



