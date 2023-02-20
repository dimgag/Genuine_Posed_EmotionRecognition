import torch
import torch.nn as nn
from collections import OrderedDict
# $ pip install timesformer-pytorch
from timesformer_pytorch import TimeSformer


# model = TimeSformer(
#     dim = 512,
#     image_size = 224,
#     patch_size = 16,
#     num_frames = 8,
#     num_classes = 10,
#     depth = 12,
#     heads = 8,
#     dim_head =  64,
#     attn_dropout = 0.1,
#     ff_dropout = 0.1
# )

# video = torch.randn(2, 8, 3, 224, 224) # (batch x frames x channels x height x width)
# mask = torch.ones(2, 8).bool() # (batch x frame) - use a mask if there are variable length videos in the same batch

# pred = model(video, mask = mask) # (2, 10)


# Ideally, I would like to use the TimeSformer model as a backbone for a multi-task learning model.
# I would like to use the TimeSformer model as a backbone for a multi-task learning model.

class MultiTaskTimeSformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = TimeSformer(dim = 512,
                                image_size = 224,
                                patch_size = 16,
                                num_frames = 8,
                                num_classes = 10,
                                depth = 12,
                                heads = 8,
                                dim_head =  64,
                                attn_dropout = 0.1,
                                ff_dropout = 0.1)
        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Identity()
        self.net.fc1 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('final', nn.Linear(self.n_features, 1))]))
        self.net.fc2 = nn.Sequential(OrderedDict([('linear', nn.Linear(self.n_features,self.n_features)),('relu1', nn.ReLU()),('final', nn.Linear(self.n_features, 6))]))

    def forward(self, x):
        real_fake_head = self.net.fc1(self.net(x))
        emotion_head = self.net.fc2(self.net(x))
        return real_fake_head, emotion_head


model = MultiTaskTimeSformer()

# print model parameters and their sizes
for name, param in model.named_parameters():
    print(name, param.size())

