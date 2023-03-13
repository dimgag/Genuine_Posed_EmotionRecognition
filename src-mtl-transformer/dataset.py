import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


# class MTL_VideoDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform

#         # Get the list of class folders
#         self.class_folders = os.listdir(self.root_dir)

#         # Initialize the lists of classes for each task
#         self.rf_classes = ['fake', 'real']
#         self.emo_classes = ['angry', 'contempt', 'disgust', 'happy', 'sad', 'surprise']

#     def __len__(self):
#         # Count the total number of videos in the dataset
#         return sum([len(os.listdir(os.path.join(self.root_dir, c))) for c in self.class_folders])

#     def __getitem__(self, idx):
#         # Select a random video and its corresponding class folder
#         video_folder_idx = np.random.randint(len(self.class_folders))
#         video_folder = self.class_folders[video_folder_idx]
#         video_path = os.path.join(self.root_dir, video_folder)
#         video_files = os.listdir(video_path)
#         video_file_idx = np.random.randint(len(video_files))
#         video_file = video_files[video_file_idx]

#         # Load the video frames
#         cap = cv2.VideoCapture(os.path.join(video_path, video_file))
#         frames = []
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
#             face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             gray = cv2.GaussianBlur(frame, (25,25), 0)
#             face = face_cascade.detectMultiScale(gray, 1.1, 4)
#             for (x,y,w,h) in face:
#                 frame = frame[y:y+h, x:x+w]
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             if self.transform:
#                 frame = self.transform(frame) #this outpus as np array. -> Convert the numpy array to tensor ?
                
#             frames.append(frame)

#         # Get the labels for the two tasks
#         rf_label = self.rf_classes.index(video_folder.split("_")[0])
#         emo_label = self.emo_classes.index(video_folder.split("_")[1])
        
#         frames = [torch.from_numpy(frame) for frame in frames]
        
        
#         frames = torch.stack(frames) # Stack is a sequence of frames.

#         sample = {'frames': frames, 'rf_label': rf_label, 'emo_label': emo_label}
#         # return torch.stack(frames), rf_label, emo_label

        
#         return sample

def collate_fn(batch):
    # get the maximum size of the tensors in the batch
    max_size = tuple(max(sample.size(i) for sample in batch) for i in range(1, len(batch[0].size())))
    
    # pad the tensors to the maximum size
    padded_batch = [torch.nn.functional.pad(sample, (0, max_size[-1] - sample.shape[-1], 0, max_size[-2] - sample.shape[-2])) for sample in batch]
    
    # stack the tensors
    return torch.stack(padded_batch, dim=0)


class VideoDataset2(Dataset):
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
                frame = cv2.resize(frame, (256, 256)) # Resize the frame

            if self.transform:
                frame = self.transform(frame) # Convert the numpy array to tensor
                
            frames.append(frame)

        # Get the labels for the two tasks
        rf_label = self.rf_classes.index(video_folder.split("_")[0])
        emo_label = self.emo_classes.index(video_folder.split("_")[1])
        
        frames = [torch.from_numpy(frame) for frame in frames]

        collate_fn(frames)         
        
        frames = torch.stack(frames) # Stack is a sequence of frames.

        sample = {'frames': frames, 'rf_label': rf_label, 'emo_label': emo_label}

        return sample




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Modify the class to extract a standard number of frames from each video.
#



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

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        n_frames = 16 # Change this to the desired number of frames

        if total_frames < n_frames:
            # Skip videos with fewer frames than the desired number
            return self.__getitem__(np.random.randint(self.__len__()))

        step_size = total_frames // n_frames
        
        frames = []
        for i in range(0, total_frames, step_size):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
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
                frame = cv2.resize(frame, (256, 256)) # Resize the frame

            if self.transform:
                frame = self.transform(frame) # Convert the numpy array to tensor

            frames.append(frame)

        # Get the labels for the two tasks
        rf_label = self.rf_classes.index(video_folder.split("_")[0])
        emo_label = self.emo_classes.index(video_folder.split("_")[1])

        frames = [torch.from_numpy(frame) for frame in frames]
        
        collate_fn(frames) 
        
        frames = torch.stack(frames) # Stack is a sequence of frames.

        sample = {'frames': frames, 'rf_label': rf_label, 'emo_label': emo_label}

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
    frames, label1, label2 = train_dataset[0]['frames'], train_dataset[0]['rf_label'], train_dataset[0]['emo_label']

    # Print the labels
    print(f"Label 1: {label1}")
    print(f"Label 2: {label2}")

    # Plot the frames
    fig, ax = plt.subplots(1, 16)
    for i in range(16):
        ax[i].imshow(frames[i])
        ax[i].axis('off')
    plt.show()


