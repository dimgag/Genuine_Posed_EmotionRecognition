# video_classifier_main.py
import sys
from pytorchvideo.data import LabeledVideoDataset
from glob import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import torch.nn as nn
import torch
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report
import torchmetrics

# Augmentation process
from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler, labeled_video_dataset
from torch.utils.data import DataLoader

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,

    UniformTemporalSubsample,
    Permute
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize
)

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)


#################### 

# Define video transforms
video_transform = Compose([
    ApplyTransformToKey(key='video',
        transform = Compose([
        UniformTemporalSubsample(20),
        Lambda(lambda x:x/255),
        NormalizeVideo(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        RandomShortSideScale(min_size=256, max_size=256),
        CenterCropVideo(224),
        RandomHorizontalFlip(p=0.5)
    ]),
    ),
])

# Define model:
# model.py
class OurModel(LightningModule):
    def __init__(self):
        super(OurModel, self).__init__()
        # Model Architecture
        self.video_model = torch.hub.load('facebookresearch/pytorchvideo', 'efficient_x3d_xs', pretrained=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(400, 12)
        
        self.lr = 1e-3
        
        self.batch_size = 8
        self.numworker=4 # If decrease it will be slower computation time
        
        # Evaluation Metric
        self.metric = torchmetrics.Accuracy(task='multiclass',   num_classes=12)
        
        # Loss Function
        self.criterion = nn.BCEWithLogitsLoss()
        

    def forward(self, x):
        x = self.video_model(x)
        x = self.relu(x)
        x = self.linear(x)
        return x

    def configure_optimizers(self):
        opt = torch.optim.AdamW(params = self.parameters(), lr = self.lr)
        scheduler = CosineAnnealingLR(opt, T_max=10, eta_min=1e-6, last_epoch=-1)
        return {'optimizer': opt, 'lr_scheduler': scheduler}


    def train_dataloader(self):
        dataset = labeled_video_dataset('../data_temporal/train_root',
                                      clip_sampler=make_clip_sampler('random', 2),
                                        transform = video_transform, decode_audio=False)


        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.numworker, pin_memory=True)
        return loader
    
    def training_step(self, batch, batch_idx):
        video, label = batch['video'], batch['label']
        out = self(video) # or self.forward(video)
        loss = self.criterion(out, label)
        metric = self.metric(out, label.to(torch.int64))
        return {'loss': loss, 'metric': metric.detach()}
    
    
    def on_train_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean().cpu().numpy().round(2)
        metric = torch.stack([x['metric'] for x in outputs]).mean().cpu().numpy().round(2)
        self.log('training_loss', loss)
        self.log('trainng_metric', metric)
    
    
    def val_dataloader(self):
        dataset = labeled_video_dataset('../data_temporal/val_root',
                                      clip_sampler=make_clip_sampler('random', 2),
                                        transform = video_transform, decode_audio=False)


        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.numworker, pin_memory=True)
        return loader
    
    def validation_step(self, batch, batch_idx):
        video, label = batch['video'], batch['label']
        out = self(video) # or self.forward(video)
        loss = self.criterion(out, label)
        metric = self.metric(out, label.to(torch.int64))
        return {'loss': loss, 'metric': metric.detach()}
    
    def on_validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean().cpu().numpy().round(2)
        metric = torch.stack([x['metric'] for x in outputs]).mean().cpu().numpy().round(2)
        self.log('val_loss', loss)
        self.log('val_metric', metric)
        
    def test_dataloader(self):
        dataset = labeled_video_dataset('../data_temporal/val_root',
                                      clip_sampler=make_clip_sampler('random', 2),
                                        transform = video_transform, decode_audio=False)


        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.numworker, pin_memory=True)
        return loader
    
    def test_step(self, batch, batch_idx):
        video, label = batch['video'], batch['label']
        out = self(video) # or self.forward(video)
        return {'label': label.detach(), 'pred': out.detach()}
    
    def on_test_epoch_end(self, outputs):
        label = torch.cat([x['label'] for x in outputs]).cpu().numpy()
        pred = torch.cat([x['pred'] for x in outputs]).cpu().numpy()
        pred = np.where(pred>0.5,1,0)
        print(classification_report(label, pred))
        
    
    
    


def main():
	# Define Checkpointer
	checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath = 'checkpoints',
	                                      filename='file', save_last=True)

	lr_monitor = LearningRateMonitor(logging_interval='epoch')

	# 10 (total), 5(no improved), 7(interupted), 7(resume)

	model = OurModel()
	seed_everything(0)

	trainer = Trainer(max_epochs = 15,
	                  accelerator = 'gpu', devices = -1,
	                      precision = 16,
	                  accumulate_grad_batches = 2,
	                  enable_progress_bar = False,
	                  num_sanity_val_steps = 0,
	                  callbacks = [lr_monitor, checkpoint_callback],
	                  
	                 )
	                  

	trainer.fit(model)
	        