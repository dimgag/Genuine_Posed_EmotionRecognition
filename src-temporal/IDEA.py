# https://github.com/fcakyon/video-transformers
# Try this implemenation for the TimeSformer model
# Install:
# conda install pytorch=1.11.0 torchvision=0.12.0 cudatoolkit=11.3 -c pytorch
# pip install git+https://github.com/facebookresearch/pytorchvideo.git
# pip install git+https://github.com/huggingface/transformers.git
# pip install video-transformers

# Prepare video classification dataset in such folder structure (.avi and .mp4 extensions are supported):
# train_root
#     label_1
#         video_1
#         video_2
#         ...
#     label_2
#         video_1
#         video_2
#         ...
#     ...
# val_root
#     label_1
#         video_1
#         video_2
#         ...
#     label_2
#         video_1
#         video_2
#         ...
#     ...

# Fine-tune Timesformer (from HuggingFace) video classifier:
from torch.optim import AdamW
from video_transformers import VideoModel
from video_transformers.backbones.transformers import TransformersBackbone
from video_transformers.data import VideoDataModule
from video_transformers.heads import LinearHead
from video_transformers.trainer import trainer_factory
from video_transformers.utils.file import download_ucf6

backbone = TransformersBackbone("facebook/timesformer-base-finetuned-k400", num_unfrozen_stages=1)

download_ucf6("./")
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

optimizer = AdamW(model.parameters(), lr=1e-4)

Trainer = trainer_factory("single_label_classification")
trainer = Trainer(datamodule, model, optimizer=optimizer, max_epochs=8)

trainer.fit()

# Perform prediction for a single file or folder of videos:
from video_transformers import VideoModel

model_name_or_path = 'facebook/timesformer-base-finetuned-k400' # this is an example, you can use any model from HuggingFace

model = VideoModel.from_pretrained(model_name_or_path)

model.predict(video_or_folder_path="video.mp4")
# [{'filename': "video.mp4", 'predictions': {'class1': 0.98, 'class2': 0.02}}]