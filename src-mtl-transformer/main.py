'''
MAIN IDEA:

I want to implement a Multi-Task Learning Transformer model with temporal information for Image Classification.
The model should be able to predict the following tasks:
Classify the emotion of a person in a given video
Classify if the emotion is real or fake

For that purpose I have the following dataset:
data_temporal /
  train_root/
           /fake_angry
           /fake_contempt
           /fake_disgust
           /fake_happy
           /fake_sad
           /fake_surprise
           /real_angry
           /real_contempt
           /real_disgust
           /real_happy
           /real_sad
           /real_surprise
  val_root/
           /fake_angry
           /fake_contempt
           /fake_disgust
           /fake_happy
           /fake_sad
           /fake_surprise
           /real_angry
           /real_contempt
           /real_disgust
           /real_happy
           /real_sad
           /real_surprise
Where each folder contains videos.

'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from tqdm.auto import tqdm
# import modules
from dataset import MTL_VideoDataset, load_mtl_dataset
from trainer import train



class MultiTaskTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim1, output_dim2, num_heads, num_layers):
        super(MultiTaskTransformer, self).__init__()
        self.d_model = input_dim
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(input_dim, num_heads, hidden_dim), num_layers)
        self.fc1 = nn.Linear(input_dim, output_dim1)
        self.fc2 = nn.Linear(input_dim, output_dim2)

    def forward(self, x):
        x = self.transformer(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return x1, x2



# Main function to run the code from the terminal
def main():
    train_dataloader, test_dataloader = load_mtl_dataset('data_temporal', batch_size=batch_size)
    print("---------------------------------------------------")
    print("Number of training batches: ", len(train_dataloader))
    print("Number of test batches: ", len(test_dataloader))
    print("---------------------------------------------------")
    # Model, loss and optimizer
    model = MultiTaskTransformer(input_dim, hidden_dim, output_dim1, output_dim2, num_heads, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epoch_loss, epoch_acc_emotion, epoch_acc_real_fake, overall_training_acc = train(model, train_dataloader, test_dataloader, optimizer, num_epochs)




if __name__ == '__main__':
  # Set the Hyperparameters
  input_dim = 3 # RGB channels
  hidden_dim = 128
  output_dim1 = 6 # 6 classes for task 1
  output_dim2 = 2 # 2 classes for task 2
  num_heads = 3
  num_layers = 4
  batch_size = 3
  lr = 0.001
  num_epochs = 10

  main()
  