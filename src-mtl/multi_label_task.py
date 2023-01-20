# Multi-Task Classification Approach
# Task 1: Classification of the emotion
# Task 2: Classification of real or fake
# 

# Data should be in the following format:
# data
# ├── train
# │   ├── real_happy_1.jpg
# │   ├── real_sad_1.jpg
# │   ├── real_angry_1.jpg
# │   ├── real_surprise_1.jpg
# │   ├── real_contempt_1.jpg
# │   ├── real_disgust_1.jpg
# │   ├── fake_happy_1.jpg
# │   ├── fake_sad_1.jpg
# │   ├── fake_angry_1.jpg
# │   ├── fake_surprise_1.jpg
# │   ├── fake_contempt_1.jpg
# │   └── fake_disgust_1.jpg
# │ 
# └── val
#     ├── real_happy_1.jpg
#     ├── real_sad_1.jpg
#     ├── real_angry_1.jpg
#     ├── real_surprise_1.jpg
#     ├── real_contempt_1.jpg
#     ├── real_disgust_1.jpg
#     ├── fake_happy_1.jpg
#     ├── fake_sad_1.jpg
#     ├── fake_angry_1.jpg
#     ├── fake_surprise_1.jpg
#     ├── fake_contempt_1.jpg
#     └── fake_disgust_1.jpg

# File names:
# [real/fake]_[emotion]_[frame_number].jpg


import torch
import PIL
import numpy as np
from torchvision import datasets, transforms


# multitaskds.py
class MultiTaskDataset(Dataset):
    def __init__(self,df, tfms, size=64):
        self.paths = list(df.name)
        self.labels = list(df.label)
        self.tfms = tfms
        self.size = size
        self.norm = transforms.Normalize([0.4270, 0.3508, 0.2971], [0.1844, 0.1809, 0.1545]) # Stats from function: get_mean_and_std(), src/data.py

    def __len__(self): return len(self.paths)

    def __getitem__(self,idx):
        #dealing with the image
        img = PIL.Image.open(self.paths[idx]).convert('RGB')
        img = Image(pil2tensor(img, dtype=np.float32).div_(255))
        img = img.apply_tfms(self.tfms, size = self.size)
        img = self.norm(img.data)

        #dealing with the labels
        labels = self.labels[idx].split(" ")
        age = torch.tensor(float(labels[0]), dtype=torch.float32)
        gender = torch.tensor(int(labels[1]), dtype=torch.int64)
        ethnicity = torch.tensor(int(labels[2]), dtype=torch.int64)
        
        return img.data, (age.log_()/4.75, gender, ethnicity)

# dsdldb.py
tfms = get_transforms()
train_ds = MultiTaskDataset(df_train, tfms[0], size=64)
valid_ds = MultiTaskDataset(df_valid, tfms[1], size=64)
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
valid_dl = DataLoader(valid_ds, batch_size=128, shuffle=True, num_workers=2)
data = DataBunch(train_dl, valid_dl)


# model.py
class MultiTaskModel(nn.Module):
    """
    Creates a MTL model with the encoder from "arch" and with dropout multiplier ps.
    """
    def __init__(self, arch,ps=0.5):
        super(MultiTaskModel,self).__init__()
        self.encoder = create_body(arch)        #fastai function that creates an encoder given an architecture
        self.fc1 = create_head(1024,1,ps=ps)    #fastai function that creates a head
        self.fc2 = create_head(1024,2,ps=ps)
        self.fc3 = create_head(1024,5,ps=ps)

    def forward(self,x):

        x = self.encoder(x)
        age = torch.sigmoid(self.fc1(x))
        gender = self.fc2(x)
        ethnicity = self.fc3(x)

        return [age, gender, ethnicity]



# multitaskloss.py
class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, preds, age, gender, ethnicity):

        mse, crossEntropy = MSELossFlat(), CrossEntropyFlat()

        loss0 = mse(preds[0], age)
        loss1 = crossEntropy(preds[1],gender)
        loss2 = crossEntropy(preds[2],ethnicity)

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1 + self.log_vars[1]

        precision2 = torch.exp(-self.log_vars[2])
        loss2 = precision2*loss2 + self.log_vars[2]
        
        return loss0+loss1+loss2

# learn_def.py
def rmse_age(preds, age, gender, ethnicity): return root_mean_squared_error(preds[0],age)
def acc_gender(preds, age, gender, ethnicity): return accuracy(preds[1], gender)
def acc_ethnicity(preds, age, gender, ethnicity): return accuracy(preds[2], ethnicity)
metrics = [rmse_age, acc_gender, acc_ethnicity]

model = MultiTaskModel(models.resnet34, ps=0.25)

loss_func = MultiTaskLossWrapper(3).to(data.device) #just making sure the loss is on the gpu

learn = Learner(data, model, loss_func=loss_func, callback_fns=ShowGraph, metrics=metrics)

#spliting the model so that I can use discriminative learning rates
learn.split([learn.model.encoder[:6],
             learn.model.encoder[6:],
             nn.ModuleList([learn.model.fc1, learn.model.fc2, learn.model.fc3])]);

#first I'll train only the last layer group (the heads)
learn.freeze()