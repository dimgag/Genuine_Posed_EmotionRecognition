# Data preprocessing - detect faces and crop the images
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pandas as pd


# Define the image path
frames = "../data/frames"
emotions = os.listdir(frames)
emotions.remove('.DS_Store')


# Find .DS_Store and remove it from every emotion folder
for emotion in emotions:
    if '.DS_Store' in os.listdir(frames + "/" + emotion):
        os.remove(frames + "/" + emotion + "/" + '.DS_Store')
        print("Removed .DS_Store from: ", emotion)



# Create a new directory for the cropped faces
cropped_faces = "../data/cropped_faces"
if not os.path.exists(cropped_faces):
    os.mkdir(cropped_faces)

# Create a new directory for each emotion if not already created
for emotion in emotions:
    if emotion not in os.listdir(cropped_faces):
        os.mkdir(cropped_faces + "/" + emotion)



# Crop the faces and save them in the new directory
for emotion in emotions:
    for image in os.listdir(frames + "/" + emotion):
        img = cv2.imread(frames + "/" + emotion + "/" + image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (25, 25), 0)
        face = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in face:
            face = img[y:y+h, x:x+w]
            cv2.imwrite(cropped_faces + "/" + emotion + "/" + image, face)

# Check the number of images in each folder
for emotion in emotions:
    print(emotion, ":", len(os.listdir(cropped_faces + "/" + emotion)))

# Remove .DS_Store from the cropped faces directory
if '.DS_Store' in os.listdir(cropped_faces):
    os.remove(cropped_faces + "/" + '.DS_Store')
    print("Removed .DS_Store from: ", cropped_faces)


