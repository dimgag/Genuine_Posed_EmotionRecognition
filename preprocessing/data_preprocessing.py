# Data preprocessing - detect faces and crop the images
# Author: Dimitrios Gagatsis
import cv2
import os

# Define the image path
frames = "data/frames"
emotions = os.listdir(frames)
emotions.remove('.DS_Store')

# Find .DS_Store and remove it from every emotion folder
for emotion in emotions:
    if '.DS_Store' in os.listdir(frames + "/" + emotion):
        os.remove(frames + "/" + emotion + "/" + '.DS_Store')
        print("Removed .DS_Store from: ", emotion)

# Create a new directory for the cropped faces
cropped_faces = "data/cropped_faces"
if not os.path.exists(cropped_faces):
    os.mkdir(cropped_faces)

# Create a new directory for each emotion if not already created
for emotion in emotions:
    if emotion not in os.listdir(cropped_faces):
        os.mkdir(cropped_faces + "/" + emotion)


# Get the Haar Cascade classifier
# Note: The haarcascade_frontalface_alt2.xml seems to work better that the haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')


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


# Remove .DS_Store from the cropped faces directory
if '.DS_Store' in os.listdir(cropped_faces):
    os.remove(cropped_faces + "/" + '.DS_Store')
    print("Removed .DS_Store from: ", cropped_faces)

# Check the number of images in each folder
for emotion in emotions:
    print(emotion, ":", len(os.listdir(cropped_faces + "/" + emotion)))

# Count the number of images in the cropped folders and those in the original folders
for emotion in emotions:
    print(emotion, "Original Images: ", len(os.listdir(frames + "/" + emotion)), "Cropped Images: ", len(os.listdir(cropped_faces + "/" + emotion)))
