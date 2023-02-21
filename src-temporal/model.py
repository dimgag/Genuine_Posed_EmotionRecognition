# To use the TimeSformer model as a baseline for multi-task learning, 
# you can modify the last layer of the model to output two logits, one for each task. 
# Here's an example of how to modify the TimeSformer model:
import torch.nn as nn
from timesformer_pytorch import TimeSformer

class MultiTaskTimeSformer1(nn.Module):
    def __init__(self, num_classes, num_tasks):
        super(MultiTaskTimeSformer, self).__init__()

        # Load the TimeSformer model
        self.timesformer = TimeSformer(
            dim = 256,
            image_size = 224,
            patch_size = 16,
            num_frames = 8,
            num_classes = num_classes
        )

        # Modify the last layer to output two logits, one for each task
        self.timesformer.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(self.timesformer.embed_dim, num_tasks)
        )

    def forward(self, x):
        return self.timesformer(x)



# # # # # # # # # # # 
import torch
import torch.nn as nn
import timm

class MultiTaskTimeSformer(nn.Module):
    def __init__(self, num_classes_real_fake, num_classes_emotions, model_name='TimeSformer_divST_16x16_448_K600.pyth', pretrained=True):

        
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features

        self.head_real_fake = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(),
            nn.Linear(self.num_features, num_classes_real_fake),

            
        )

        self.head_emotions = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(),
            nn.Linear(self.num_features, num_classes_emotions),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x[:, -1, :]
        out_fake = self.head_real_fake(x)
        out_emotion = self.head_emotions(x)
        return out_fake, out_emotion


# model = MultiTaskTimeSformer(num_classes_real_fake=2, num_classes_emotions=6)



# # print model parameters and their sizes
# for name, param in model.named_parameters():
#     print(name, param.size())

##########################################################################################################
# Huggingface Transformers library for video classification

from torch.optim import AdamW
# from video_transformers import VideoModel
from video_transformers.backbones.transformers import TransformersBackbone
from video_transformers.data import VideoDataModule
from video_transformers.heads import LinearHead
from video_transformers.trainer import trainer_factory
from video_transformers.utils.file import download_ucf6

backbone = TransformersBackbone("facebook/timesformer-base-finetuned-k400", num_unfrozen_stages=1)

# download_ucf6("./")
datamodule = VideoDataModule(
    train_root="ucf6/train",
    val_root="ucf6/val",
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

model = VideoModel(backbone, head)



# optimizer = AdamW(model.parameters(), lr=1e-4)

# Trainer = trainer_factory("single_label_classification")
# trainer = Trainer(datamodule, model, optimizer=optimizer, max_epochs=8)

# trainer.fit()



class MTLTimeSformer(nn.Module):
    def __init__(self, num_classes_real_fake, num_classes_emotions):
        super().__init__()
        self.backbone = TransformersBackbone("facebook/timesformer-base-finetuned-k400", num_unfrozen_stages=1)
        self.num_features = self.backbone.num_features

        self.head_real_fake = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(),
            nn.Linear(self.num_features, num_classes_real_fake),            
        )

        self.head_emotions = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(),
            nn.Linear(self.num_features, num_classes_emotions),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x[:, -1, :]
        out_real_fake = self.head_real_fake(x)
        out_emotions = self.head_emotions(x)
        return out_real_fake, out_emotions


model = MTLTimeSformer(num_classes_real_fake=2, num_classes_emotions=6)

# print model parameters and their sizes
for name, param in model.named_parameters():
    print(name, param.size())

