import torch
import torch.nn as nn
import json
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.optim import AdamW
from video_transformers import VideoModel
from video_transformers.backbones.transformers import TransformersBackbone
from video_transformers.data import VideoDataModule
from video_transformers.heads import LinearHead
from video_transformers.trainer import trainer_factory
from video_transformers.utils.file import download_ucf6


from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from typing import Dict



# Define the device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transforms:
# transform = Compose([
#     ApplyTransformToKey(
#         key="video",
#         transform=Compose([
#             UniformTemporalSubsample(8),
#             ShortSideScale(size=256),
#             UniformCropVideo(size=224),
#             CenterCropVideo(size=224),
#             NormalizeVideo(mean=[0.4270, 0.3508, 0.2971], std=[0.1844, 0.1809, 0.1545]),
#             ToTensor(),
#         ]),
#     ),
# ])


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Here define the transforms for the videos to detect faces and crop them:
'''import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN

mtcnn = MTCNN()

# Define the transforms for the videos - this requires the MTCNN face detector and frames generated from the videos.
transform = transforms.Compose([
    # Apply MTCNN face detection and cropping
    transforms.Lambda(lambda frames: [mtcnn(frame) for frame in frames]),
    transforms.Lambda(lambda frames: [frame for frame in frames if frame is not None]),
    # Apply other transforms
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4270, 0.3508, 0.2971], std=[0.1844, 0.1809, 0.1545])
])

'''
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



# Load model from huggingface.co:
class MyNetwork(nn.Module):
    def __init__(self, num_classes):
        super(MyNetwork, self).__init__()
        self.backbone = TransformersBackbone("facebook/timesformer-base-finetuned-k400", num_unfrozen_stages=1)
        self.fc = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


backbone = TransformersBackbone("facebook/timesformer-base-finetuned-k400", num_unfrozen_stages=1)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Load the data
datamodule = VideoDataModule(
    train_root="data_temporal/train_root",
    val_root="data_temporal/val_root",
    batch_size=4,
    num_workers=4,
    num_timesteps=8,
    preprocess_input_size=224,
    preprocess_clip_duration=1,
    preprocess_means=backbone.mean,
    preprocess_stds=backbone.std,
    preprocess_min_short_side=256,
    preprocess_max_short_side=320,
    preprocess_horizontal_flip_p=0.5,
)




head = LinearHead(hidden_size=backbone.num_features, num_classes=datamodule.num_classes)
# model = MyNetwork(num_classes=12).to(device)
model = VideoModel(backbone, head)
#
optimizer = AdamW(model.parameters(), lr=1e-4)
Trainer = trainer_factory("single_label_classification")
trainer = Trainer(datamodule, model, optimizer=optimizer, max_epochs=8)
trainer.fit()
#
