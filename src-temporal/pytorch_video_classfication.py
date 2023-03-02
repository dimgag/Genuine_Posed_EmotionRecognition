import torch
from torch.optim import AdamW
from video_transformers import VideoModel
from video_transformers.backbones.transformers import TransformersBackbone
from video_transformers.data import VideoDataModule
from video_transformers.heads import LinearHead
from video_transformers.trainer import trainer_factory
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample, UniformCropVideo


# Define the device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = TransformersBackbone("facebook/timesformer-base-finetuned-k400", num_unfrozen_stages=1)

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

# Define model Head
head = LinearHead(hidden_size=backbone.num_features, num_classes=datamodule.num_classes)

# Compile the model
model = VideoModel(backbone, head)
optimizer = AdamW(model.parameters(), lr=1e-4)

# Set trainer
Trainer = trainer_factory("single_label_classification")
trainer = Trainer(datamodule, model, optimizer=optimizer, max_epochs=30, gpus=1, )
trainer.fit()
