'''# Live Camera using opencv dnn face detection & Emotion Recognition
#
import enum
import sys
import time
import argparse
import cv2
import numpy as np
import torch


# sys.path.insert(1, 'face_detector')

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
    t2 = time.time()
    print('frame.shape:', frame.shape)
    print('t2-t1:', t2-t1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# -----------------------------------------------------------------------------------
# to this snippet from camera-demo/camera_demo_2.py:'''




# Live Camera using opencv dnn face detection & Emotion Recognition
#
import enum
import sys
import time
import argparse
import cv2
import numpy as np
import torch


# sys.path.insert(1, 'face_detector')

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
    t2 = time.time()
    print('frame.shape:', frame.shape)
    print('t2-t1:', t2-t1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# -----------------------------------------------------------------------------------
# to this snippet from camera-demo/camera_demo_2.py: