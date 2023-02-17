from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

class SASEFE_MTL(Dataset):
    def __init__(self, image_paths):
        # a function defining the elements of a dataset (like inputs and labels)
        # transforms.Normalize(mean=[0.4270, 0.3508, 0.2971], std=[0.1844, 0.1809, 0.1545])
        # Define Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(p=0.5), #Data Augmentation
            # transforms.RandomRotation(35), #Data Augmentation
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.4270, 0.3508, 0.2971], std=[0.1844, 0.1809, 0.1545])
            ])


        # Set Inputs and Labels
        self.image_paths = image_paths
        self.images = []
        self.real_fakes = []
        self.emotions = []

        for path in image_paths:
            filename = path[0:].split("_")
            if len(filename) == 3:
                self.images.append(path)
                self.real_fakes.append(int(filename[0]))
                self.emotions.append(int(filename[1]))


            
    def __len__(self):
        # This function just returns the number of images.
        return len(self.images)


    def __getitem__(self, index):
        # A function that returns an item from the dataset
        # Load an Image
        # print(self.images[index])
        img = Image.open("data_mtl/train/" + self.images[index]).convert('RGB')
        # Transform the image
        img = self.transform(img)

        # Get the labels
        real_fake = self.real_fakes[index]
        emotion = self.emotions[index]

        # Return the sample of the dataset
        sample = {'image': img, 'real_fake': real_fake, 'emotion': emotion}
        
        return sample



class SASEFE_MTL_TEST(Dataset):
    def __init__(self, image_paths):
        # a function defining the elements of a dataset (like inputs and labels)
        # transforms.Normalize(mean=[0.4270, 0.3508, 0.2971], std=[0.1844, 0.1809, 0.1545])
        # Define Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4270, 0.3508, 0.2971], std=[0.1844, 0.1809, 0.1545])
            ])

        # Set Inputs and Labels
        self.image_paths = image_paths
        self.images = []
        self.real_fakes = []
        self.emotions = []

        for path in image_paths:
            filename = path[0:].split("_")
            if len(filename) == 3:
                self.images.append(path)
                self.real_fakes.append(int(filename[0]))
                self.emotions.append(int(filename[1]))


            
    def __len__(self):
        # This function just returns the number of images.
        return len(self.images)


    def __getitem__(self, index):
        # A function that returns an item from the dataset
        # Load an Image
        # print(self.images[index])
        img = Image.open("data_mtl/test/" + self.images[index]).convert('RGB')
        # Transform the image
        img = self.transform(img)

        # Get the labels
        real_fake = self.real_fakes[index]
        emotion = self.emotions[index]

        # Return the sample of the dataset
        sample = {'image': img, 'real_fake': real_fake, 'emotion': emotion}
        
        return sample


################################################################################# 
# Make a dataset
# train_dir = "data_mtl/train"

# train_image_paths = os.listdir("data_mtl/train")

# train_dataloader = DataLoader(SASEFE_MTL(train_image_paths), shuffle=True, batch_size=32)


# # print(train_dataloader.__len__())


# print(train_dataloader.dataset.__getitem__(1))




