import torch
from torch.utils.data import DataLoader

from video_transformers import VideoModel
from video_transformers.backbones.transformers import TransformersBackbone
from video_transformers.data import VideoDataModule
from video_transformers.heads import LinearHead

# Define your model
backbone = TransformersBackbone("facebook/timesformer-base-finetuned-k400", num_unfrozen_stages=1)


head = LinearHead(hidden_size=backbone.num_features, num_classes=datamodule.num_classes)

model = VideoModel(backbone, head)


# Load the checkpoint
checkpoint = torch.load('runs/exp/checkpoint/pytorch_model.bin')
model.load_state_dict(checkpoint['model_state_dict'])

# Define your evaluation dataset
eval_dataset = MyVideoDataset(val_root, transform=val_transform)

# Define your evaluation data loader
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# Set the model to evaluation mode
model.eval()

# Define your evaluation metrics
correct = 0
total = 0

# Evaluate the model on the evaluation dataset
with torch.no_grad():
    for inputs, labels in eval_loader:
        # Forward pass
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # Update evaluation metrics
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate and print the evaluation accuracy
accuracy = 100 * correct / total
print('Accuracy of the network on the evaluation dataset: {} %'.format(accuracy))
