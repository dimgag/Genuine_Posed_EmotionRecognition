import os
import cv2


# TO DO:


# Define a temporal model for Video Classification


# Path: src-temporal/main.py
# Get the data directory
data_dir = "data/SASE-FE/FakeTrue_DB"

# Those files are videos
files = os.listdir(data_dir)

# Get the file paths
files_paths = [os.path.join(data_dir, file) for file in files]

# Get the file names
files_names = [file for file in files]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Models I want to try here are: 
# 1. TimeSformer (https://arxiv.org/abs/2102.05095)
# 2. DeVTR (https://ieeexplore.ieee.org/abstract/document/9530829)

# Clone the TimeSformer repo
# pip install timesformer-pytorch (IDK if this is the right way to do it)
# You can load the pretrained models as follows:
import torch
from timesformer.models.vit import TimeSformer

model = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='/path/to/pretrained/model.pyth')

dummy_video = torch.randn(2, 3, 8, 224, 224) # (batch x channels x frames x height x width)

pred = model(dummy_video,) # (2, 400)
