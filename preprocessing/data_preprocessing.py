# Data preprocessing to detect faces and prepare the data for the model
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pandas as pd


# Define the image path
classes = "../data/frames"
classes_emotions = os.listdir(classes)
classes_emotions.remove('.DS_Store')

# Count the number  of classes
print("Number of classes is: ", len(classes_emotions))


# Find .DS_Store and remove it from every emotion folder
for emotion in classes_emotions:
    if '.DS_Store' in os.listdir(classes + "/" + emotion):
        os.remove(classes + "/" + emotion + "/" + '.DS_Store')
        print("Removed .DS_Store from: ", emotion)


# Visualize the data
# get the first image of the first emotion
img = cv2.imread(classes + "/" + classes_emotions[0] + "/" + os.listdir(classes + "/" + classes_emotions[0])[0])


# Load the cascade
# Detect faces
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread(classes + "/" + classes_emotions[0] + "/" + os.listdir(classes + "/" + classes_emotions[0])[0])

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur the image background
gray = cv2.GaussianBlur(gray, (25, 25), 0)


faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the output
plt.imshow(img)
plt.axis("off")
plt.show()

# Crop the faces and save them in a new folder
# Create a new folder to save the cropped faces
cropped_faces = "../data/cropped_faces"
os.mkdir(cropped_faces)

# Create a new folder for each emotion
for emotion in classes_emotions:
    os.mkdir(cropped_faces + "/" + emotion)

'''
# Crop the faces and save them in the new folder
for emotion in classes_emotions:
    for image in os.listdir(classes + "/" + emotion):
        img = cv2.imread(classes + "/" + emotion + "/" + image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (25, 25), 0)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            cv2.imwrite(cropped_faces + "/" + emotion + "/" + image, face)
'''
# Check the number of images in each folder
for emotion in classes_emotions:
    print(emotion, ":", len(os.listdir(cropped_faces + "/" + emotion)))

# Better way to extract faces from images

# Extract faces from images
def extract_faces(emotion):
    faces = []
    for image in os.listdir(cropped_faces + "/" + emotion):
        img = cv2.imread(cropped_faces + "/" + emotion + "/" + image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (25, 25), 0)
        face = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in face:
            faces.append(gray[y:y+h, x:x+w])
    return faces


# A LOT TO DO HERE.... ^_^