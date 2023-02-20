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


#TODO: Still I have to adapt the last layer here.


# # # # # # # # # # # 
import torch
import torch.nn as nn
import timm

class MultiTaskTimeSformer(nn.Module):
    def __init__(self, num_classes_fake, num_classes_emotion, model_name='timesformer_vit_base_patch16_224', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features

        self.head_fake = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(),
            nn.Linear(self.num_features, num_classes_fake),
        )

        self.head_emotion = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(),
            nn.Linear(self.num_features, num_classes_emotion),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x[:, -1, :]
        out_fake = self.head_fake(x)
        out_emotion = self.head_emotion(x)
        return out_fake, out_emotion


model = MultiTaskTimeSformer(num_classes_fake=2, num_classes_emotion=6)

# print model parameters and their sizes
for name, param in model.named_parameters():
    print(name, param.size())
