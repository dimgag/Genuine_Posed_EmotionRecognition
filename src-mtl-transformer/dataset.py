import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class MTL_VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Get the list of class folders
        self.class_folders = os.listdir(self.root_dir)

        # Initialize the lists of classes for each task
        self.rf_classes = ['fake', 'real']
        self.emo_classes = ['angry', 'contempt', 'disgust', 'happy', 'sad', 'surprise']

    def __len__(self):
        # Count the total number of videos in the dataset
        return sum([len(os.listdir(os.path.join(self.root_dir, c))) for c in self.class_folders])

    def __getitem__(self, idx):
        # Select a random video and its corresponding class folder
        video_folder_idx = np.random.randint(len(self.class_folders))
        video_folder = self.class_folders[video_folder_idx]
        video_path = os.path.join(self.root_dir, video_folder)
        video_files = os.listdir(video_path)
        video_file_idx = np.random.randint(len(video_files))
        video_file = video_files[video_file_idx]

        # Load the video frames
        cap = cv2.VideoCapture(os.path.join(video_path, video_file))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(frame, (25,25), 0)
            face = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x,y,w,h) in face:
                frame = frame[y:y+h, x:x+w]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.transform:
                frame = self.transform(frame)
                
            frames.append(frame)

        # Get the labels for the two tasks
        rf_label = self.rf_classes.index(video_folder.split("_")[0])
        emo_label = self.emo_classes.index(video_folder.split("_")[1])

        frames = [torch.from_numpy(frame) for frame in frames]

        sample = {'frames': frames, 'rf_label': rf_label, 'emo_label': emo_label}
        # return torch.stack(frames), rf_label, emo_label

        
        return sample



class MyTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, frame):
        # Convert NumPy array to PIL Image
        frame = Image.fromarray(frame)
        # Apply transformations
        frame = transforms.Resize(self.size)(frame)
        
        # Add more transformations here if needed
        
        # Convert PIL Image back to NumPy array
        frame = np.array(frame)
        return frame

def load_mtl_dataset(data_dir, batch_size):
    # define the transform
    transform = MyTransform((256, 256))

    # define the train and test datasets
    train_dataset = MTL_VideoDataset(os.path.join(data_dir, 'train_root'), transform=transform)
    test_dataset = MTL_VideoDataset(os.path.join(data_dir, 'val_root'), transform=transform)

    # define the train and test dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_dataloader, test_dataloader


# test the code
if __name__ == "__main__":
    # define the data directory
    # data_dir = 'data_temporal/train_root'

    # define the transform
    transform = MyTransform((256, 256))

    train_dataset = MTL_VideoDataset('data_temporal/train_root', transform=transform)
    test_dataset = MTL_VideoDataset('data_temporal/val_root', transform=transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)

    print("---------------------------------------------------")
    print("Number of training videos: ", len(train_dataset))
    print("Number of test videos: ", len(test_dataset))
    print("Number of training batches: ", len(train_dataloader))
    print("Number of test batches: ", len(test_dataloader))
    print("---------------------------------------------------")

    # print(dataset[0][0].shape)
    print("---------------------------------------------------")
    print("Example output: ")
    # Get an example output with labels
    frames, label1, label2 = train_dataset[0]
    # Print the labels
    print(f"Label 1: {label1}")
    print(f"Label 2: {label2}")
    

    # plot the first 10 frames
    fig, ax = plt.subplots(2, 5)
    for i in range(10):
        ax[i//5, i%5].imshow(frames[i])
    plt.show()



