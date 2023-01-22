import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict


class HydraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(pretrained=True)
        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Identity()
        self.net.fc1 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('final', nn.Linear(self.n_features, 2))]))
        self.net.fc2 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('final', nn.Linear(self.n_features, 6))]))
        
    def forward(self, x):
        real_fake_head = self.net.fc1(self.net(x))
        emotion_head = self.net.fc2(self.net(x))
        return real_fake_head, emotion_head





# L_1: the emotion Loss, is a multi-class classification loss. In our case, it’s Cross-Entropy!
# L_2: the Real_Fake Loss, is a Binary Classification loss. In our case, Binary Cross-Entropy.






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = HydraNet(pretrained=True)
model = HydraNet(net).to(device)
emotion_loss = nn.CrossEntropyLoss() # Includes Softmax
real_fake_loss = nn.BCELoss() # Doesn't include Softmax



optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.09)
Sig = nn.Sigmoid()



# Import dataloader
from dataset import SASEFE_MTL
from torch.utils.data import DataLoader

# Import the dataset
train_dir = "data_mtl/train"
test_dir = "data_mtl/test"

train_dataset = SASEFE_MTL(train_image_paths)
test_dataset = SASEFE_MTL(test_image_paths)


# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

epochs = 10

for epoch in range(epochs):
   model.train()
   total_training_loss = 0
   for i, data in enumerate(train_dataloader):
        inputs = data["image"].to(device=device)
        

        real_fake_label = data["real_fake"].to(device=device)
        emotion_label = data["emotion"].to(device=device)
        
        optimizer.zero_grad()
        
        real_fake_output, emotion_output = model(inputs)
        
        loss_1 = emotion_loss(emotion_output, emotion_label)
        loss_2 = real_fake_loss(Sig(real_fake_output), real_fake_label.unsqueeze(1).float())

        
        loss = loss_1 + loss_2
        loss.backward()
        
        optimizer.step()
        total_training_loss += loss


# I need to add the accuracy 