# Live Camera using opencv dnn face detection & Emotion Recognition
#
import enum
import sys
import time
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from models import Net, EfficientNetV2M, FaceNet, FaceNet_withClassifier, VGGFace, VGGFace2, ViT

# Load emotion classification model
path = '/Users/dim__gag/git/Genuine_Posed_EmotionRecognition/experiments/experiments_2nd_data_split/exp6-facenet-lrscheduler/model.pth'

# model = FaceNet()
model = FaceNet_withClassifier()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loaded_checkpoint = torch.load(path, map_location=torch.device('cpu'))
model.load_state_dict(loaded_checkpoint['model_state_dict'])
optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
epoch = loaded_checkpoint['epoch']
loss = loaded_checkpoint['loss']


model.eval()

# Define emotion labels
emotion_labels = ['fake_angry', 'fake_contempt', 'fake_disgust', 'fake_happy', 'fake_sad', 'fake_surprise', 'real_angry', 'real_contempt', 'real_disgust', 'real_happy', 'real_sad', 'real_surprise']
video = cv2.VideoCapture(0) # 480, 640
isOpened = video.isOpened()
print('video.isOpened:', isOpened)

t1 = 0
t2 = 0

while isOpened:
    _, frame = video.read()
    isOpened = video.isOpened()    

    if frame is None:
        print('frame is None')
        break

    t1 = time.time()
    # frame = cv2.resize(frame, (640, 480))
    # frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (720, 480))
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Detect faces in the frame
    # face_cascade = cv2.CascadeClassifier('face_detector/haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Process each detected face
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img.astype('float32') / 255.0
        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = np.expand_dims(face_img, axis=0)

        print('face_img.shape:', face_img.shape)
        print('face_img.dtype:', face_img.dtype)

        with torch.no_grad():
            inputs = torch.from_numpy(face_img).float()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            emotion = emotion_labels[predicted.item()]

        # Draw rectangle around the detected face and display the classified emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    t2 = time.time()
    print('frame.shape:', frame.shape)
    print('t2-t1:', t2-t1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
