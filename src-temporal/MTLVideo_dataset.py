import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random

# Example videopath = 'data_temporal/FakeTrue_DB/Real/N2A.MP4'

class MultiTaskDataset(Dataset):
    def __init__(self, video_paths, transform=None):
        self.video_paths = video_paths
        self.transform = transform

        # Sef Inputs and Labels
        self.videos = []
        self.real_fake_labels = []
        self.emotion_labels = []

        for path in video_paths:
            filename = path.split('/')[-1]
            filename = filename.split('.')[0]
            filename = filename.split('_')
            if len(filename) == 2:
                self.videos.append(path)
                self.real_fake_labels.append(filename[0])
                self.emotion_labels.append(filename[1])

    def __len__(self):
        return len(self.video_paths)

    # I want to create frame sequences of 20 frames from every video with the faces cropped out.
    def __getitem__(self, idx):

        # Get the video path
        video_path = self.videos[idx]

        # Get the real/fake label
        real_fake_label = self.real_fake_labels[idx]

        # Get the emotion label
        emotion_label = self.emotion_labels[idx]

        # Open the video
        cap = cv2.VideoCapture(video_path)

        # Get the number of frames in the video
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get the frame rate
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

        # Get the width and height of the video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create an empty array to store the frames
        frames = np.empty((num_frames, height, width, 3), np.dtype('uint8'))

        # Read the frames
        fc = 0
        ret = True

        while (fc < num_frames  and ret):
            ret, frames[fc] = cap.read()
            fc += 1

        # Close the video
        cap.release()

        # Get the number of frames in the video
        num_frames = frames.shape[0]

        # Create an empty array to store the cropped faces
        cropped_faces = np.empty((num_frames, 224, 224, 3), np.dtype('uint8'))

        # Create a face detector
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # Loop through the frames and crop the faces
        for i in range(num_frames):
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            # Detect the faces
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            # If there are no faces, return the frame
            if len(faces) == 0:
                cropped_faces[i] = frames[i]
            # If there is one face, crop the face and return the frame
            elif len(faces) == 1:
                x, y, w, h = faces[0]
                cropped_faces[i] = frames[i][y:y+h, x:x+w]
            # If there are multiple faces, crop the face with the largest area and return the frame 
            else:
                areas = []
                for face in faces:
                    x, y, w, h = face
                    areas.append(w*h)
                index = areas.index(max(areas))
                x, y, w, h = faces[index]
                cropped_faces[i] = frames[i][y:y+h, x:x+w]

        # Create an empty array to store the frame sequences
        frame_sequences = np.empty((num_frames-19, 20, 224, 224, 3), np.dtype('uint8'))

        # Create the frame sequences
        for i in range(num_frames-19):
            frame_sequences[i] = cropped_faces[i:i+20]

        # Convert the frame sequences to a tensor
        frame_sequences = torch.from_numpy(frame_sequences)

        # Convert the real/fake label to a tensor
        if real_fake_label == 'Real':
            real_fake_label = torch.tensor([1, 0])
        else:
            real_fake_label = torch.tensor([0, 1])

        # Convert the emotion label to a tensor
        if emotion_label == 'A':
            emotion_label = torch.tensor([1, 0, 0, 0, 0, 0])
        elif emotion_label == 'C':
            emotion_label = torch.tensor([0, 1, 0, 0, 0, 0])
        elif emotion_label == 'D':
            emotion_label = torch.tensor([0, 0, 1, 0, 0, 0])
        elif emotion_label == 'H':
            emotion_label = torch.tensor([0, 0, 0, 1, 0, 0])
        elif emotion_label == 'S':
            emotion_label = torch.tensor([0, 0, 0, 0, 1, 0])
        else:
            emotion_label = torch.tensor([0, 0, 0, 0, 0, 1])

        # Return the frame sequences, real/fake label, and emotion label
        return frame_sequences, real_fake_label, emotion_label



