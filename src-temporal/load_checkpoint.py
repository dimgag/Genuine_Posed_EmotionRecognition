import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from video_transformers import VideoModel
from video_transformers.backbones.transformers import TransformersBackbone
from video_transformers.data import VideoDataModule
from video_transformers.heads import LinearHead


# Define your backbone
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

# Define your model
head = LinearHead(hidden_size=backbone.num_features, num_classes=datamodule.num_classes)
model = VideoModel(backbone, head)

# Define your optimizer
optimizer = AdamW(model.parameters(), lr=1e-4)

# Define your loss function
criterion = nn.CrossEntropyLoss()

# Load the checkpoint
checkpoint = torch.load('runs/exp/checkpoint/pytorch_model.bin', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# Number of epochs
# num_epochs = 10

# # Continue training from the loaded checkpoint
# for epoch in range(epoch, num_epochs):
#     for i, (inputs, labels) in enumerate(train_loader):
#         # Forward pass
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if (i+1) % 10 == 0:
#             print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

#     # Save checkpoint after each epoch
#     torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#             }, 'runs/exp/checkpoint/pytorch_model_epoch{}.bin'.format(epoch+1))
