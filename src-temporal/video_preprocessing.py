# Data pre-processing on videos:
# define the transforms for the videos to detect faces and crop them:
# To be continued...
import torch
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

