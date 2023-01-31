# Create a class for the video dataset

# This class will be used to load the video frames and apply the transformations

# NOT READY YET... THIS IS THE FIRST IDEA
import cv2 
import os
import numpy as np
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, video_path, transform=None):
        self.video_path = video_path
        self.transform = transform
        self.video = self.load_video()
        self.num_frames = len(self.video)
        self.frame_indices = list(range(self.num_frames))
        self.frame_indices = self.frame_indices[::2]
        self.num_frames = len(self.frame_indices)
        self.video = self.video[self.frame_indices]
        self.video = self.video.astype(np.float32)
        self.video = self.video / 255.0

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        frame = self.video[idx]
        if self.transform:
            frame = self.transform(frame)
        return frame

    def load_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames


# Call the class:
participants = 'data/SASE-FE/FakeTrue_DB'

participants = list(os.listdir(participants))

for participant in participants:
    video_paths = 'data/SASE-FE/FakeTrue_DB/' + participant
    dataset = VideoDataset(video_paths)


# Think about that... maybe I have to separate the videos in folders with emotions.


