import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

class SASEFE_MTL(Dataset):
    def __init__(self, image_paths):
        # a function defining the elements of a dataset (like inputs and labels)
        # transforms.Normalize(mean=[0.4270, 0.3508, 0.2971], std=[0.1844, 0.1809, 0.1545])
        # Define Transforms
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4270, 0.3508, 0.2971], std=[0.1844, 0.1809, 0.1545])
            ])

        # Set Inputs and Labels
        self.image_paths = image_paths
        self.images = []
        self.real_fake = []
        self.emotion = []

        for path in image_paths:
            filename = path[8:].split("_")
            if len(filename) == 2:
                self.images.append(path)
                self.real_fake.append(str(filename[0]))
                self.emotion.append(str(filename[1]))


            
    def __len__(self):
        # This function just returns the number of images.
        return len(self.images)


    def __getitem__(self, index):
        # A function that returns an item from the dataset
        # Load an Image
        img = Image.open(self.images[index]).convert('RGB')
        # Transform the image
        img = self.transform(img)

        # Get the labels
        real_fake = self.real_fake[index]
        emotion = self.emotion[index]

        # Return the sample of the dataset
        sample = {'image': img, 'real_fake': real_fake, 'emotion': emotion}
        return sample
