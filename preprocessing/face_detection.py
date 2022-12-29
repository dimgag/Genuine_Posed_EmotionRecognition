# Data Preprocessing to detect faces
import matplotlib.pyplot as plt
import cv2
import os

# Define the image path
real = "data/frames_6_labels/real"
real_emotions = os.listdir(real)

# get the first image of the first emotion
img = cv2.imread(real + "/" + real_emotions[0] + "/" + os.listdir(real + "/" + real_emotions[0])[0])


# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread(real + "/" + real_emotions[0] + "/" + os.listdir(real + "/" + real_emotions[0])[0])

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur the image background
gray = cv2.GaussianBlur(gray, (25, 25), 0)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the output
plt.imshow(img)
plt.axis("off")
plt.show()
