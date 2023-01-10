# Data preprocessing - detect faces and crop the images 
# Author: Dimitrios Gagatsis
import cv2
import os
import shutil


''' Create a Crop faces function that detect faces from frames and crop them
Add this function to preprocessing utils later on'''
def crop_faces(folder):
    emotions = os.listdir(folder)


    if not os.path.exists(folder + "/" + "cropped_faces"):
        os.mkdir(folder + "/" + "cropped_faces")

    cropped_faces = folder + "/" + "cropped_faces"

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
        for image in os.listdir(folder + "/" + emotion):
            img = cv2.imread(folder + "/" + emotion + "/" + image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (25, 25), 0)
            face = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in face:
                face = img[y:y+h, x:x+w]
                cv2.imwrite(cropped_faces + "/" + emotion + "/" + image, face)

    print("Faces cropped and saved in: ", cropped_faces)

    # Clear the directory
    for emotion in os.listdir(folder):
        os.rmtree(folder + "/" + emotion)
    
    # Move emotions from cropped_faces to folder
    for emotion in os.listdir(cropped_faces):
        shutil.move(cropped_faces + "/" + emotion, folder + "/" + emotion)
    
    os.rmdir(cropped_faces)




# Add this to new_dirs_os.py
# from preprocessing_utils import crop_faces

folder = "data/train_prep"
crop_faces(folder)


# folder = "data/test_prep"
# crop_faces(folder)
