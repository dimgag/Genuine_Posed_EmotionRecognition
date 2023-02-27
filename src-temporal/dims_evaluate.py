import torch
from torch.utils.data import DataLoader

from video_transformers import VideoModel
from video_transformers.backbones.transformers import TransformersBackbone
from video_transformers.data import VideoDataModule
from video_transformers.heads import LinearHead



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your model
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

head = LinearHead(hidden_size=backbone.num_features, num_classes=datamodule.num_classes)

model = VideoModel(backbone, head)


# Load the checkpoint

checkpoint = torch.load('runs/exp/checkpoint/pytorch_model.bin', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Define your evaluation dataset
eval_dataset = datamodule.val_dataset

# Set the model to evaluation mode
model.eval()

# Define your evaluation metrics
correct = 0
total = 0

# Evaluate the model on the evaluation dataset
with torch.no_grad():
    for inputs, labels in eval_dataset:
        # Forward pass
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # Update evaluation metrics
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate and print the evaluation accuracy
accuracy = 100 * correct / total
print('Accuracy of the network on the evaluation dataset: {} %'.format(accuracy))
