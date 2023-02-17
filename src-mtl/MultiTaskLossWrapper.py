import torch
import torch.nn as nn

# idea1: Combine the two losses into one loss function based on the precision scores. 

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, preds, age, gender, ethnicity):# Edit the inputs to fit your needs....... real_fake_label, emotion_label):
        loss_real_fake = nn.CrossEntropyLoss()
        loss_emotions = nn.CrossEntropyLoss()

        ########################################################################
        # Calculate the losses, as you did in the original code
        # Edit this code ? to make it work somehow... after the calculation of the los

        crossEntropy = nn.CrossEntropyLoss()

        loss_real_fake = crossEntropy(real_fake_prediction, real_fake_label) # To adjust. I have to add this to the training loop. train.py (and validation loop)
        
        loss_emotions = crossEntropy(emotion_prediction, emotio_label) # To adjust.

        precision_real_fake = torch.exp(-self.log_vars[0])
        loss_real_fake = precision_real_fake * loss_real_fake + self.log_vars[0]

        precision_emotions = torch.exp(-self.log_vars[1])
        loss_emotions = precision_emotions * loss_emotions + self.log_vars[1]
		
        return loss_real_fake + loss_emotions










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


# Function to calculate accuracy 

def multitask_accuracy(outputs, labels):
	_, preds_real_fake = torch.max(outputs[0], 1) # I might need to adjust this
	_, preds_emotions  = torch.max(outputs[1], 1) # I might need to adjust this

	running_corrects = torch.sum(preds_real_fake == labels[:,0].data) # When the first accuracy is calculated, it is based on the real_fake prediction.
	running_corrects += torch.sum(preds_emotions == labels[:,1].data) # When the second accuracy is calculated, it is based on the emotion prediction.
    # The calculation of runnint_corrects is based on the labels. If the first prediction is correct, then the first label is correct. 
    # If the second prediction is correct, then the second label is correct.
    #  Thus, count this as an overall correct prediction.

	return running_corrects

'''Sure! This is a function called multitask_accuracy that takes in two arguments: outputs and labels. It appears to be designed to calculate the accuracy of a multi-task learning model that predicts two different things at once.
The outputs argument is expected to be a list of tensors, where each tensor represents the predicted output for one of the tasks. Specifically, the first tensor in the list represents the predicted output for a binary classification task (real vs fake), while the second tensor represents the predicted output for a multi-class classification task (emotions).
The labels argument is also expected to be a tensor, where each row corresponds to the labels for one example, and the first column represents the true labels for the binary classification task, and the second column represents the true labels for the multi-class classification task.
The function then proceeds to calculate the accuracy of the model by comparing the predicted outputs to the true labels. First, it uses the torch.max function to get the index of the maximum value along the first dimension of each tensor in outputs. This gives us the predicted label for each example.
Next, it compares the predicted labels for each task to the true labels in labels using the == operator. It adds up the number of correct predictions for each task using the torch.sum function and returns the total number of correct predictions.
Note that this function does not return the accuracy as a percentage or decimal value, but rather the total number of correct predictions. To calculate the accuracy as a percentage, you would need to divide this value by the total number of examples and multiply by 100.'''



# Weighted loss function with custom weights - need experimentation for the weights.
def multi_task_loss(outputs, target):
    cross_entropy = nn.CrossEntropyLoss()

    target_real_fake, target_emotion = target[:,0], target[:,1]
    output_real_fake, output_emotion = outputs[0], outputs[1]

    loss_real_fake = cross_entropy(output_real_fake, target_real_fake.long())
    loss_emotions = cross_entropy(output_emotion, target_emotion.long())

    # the .16 and .44 is a result of experimenting different values 
    # The real_fake in this case has a smaller weight than emotions in the loss result.
    # Doesn't feel like the best way to combine loss functions.

    return loss_real_fake/(.16) + loss_emotions/(.44)

