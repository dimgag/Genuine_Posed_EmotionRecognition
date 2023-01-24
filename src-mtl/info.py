# Convert real fake label to 0 and 1 
import os
import numpy as np

train_dir = "data_mtl/train"
Test_dir = "data_mtl/test"

train_image_paths = os.listdir(train_dir)
test_image_paths = os.listdir(Test_dir)


# I want to rename the files in the test folder to 0 and 1

# A file name looks like this:
# real_angry_H2N2A.MP4age103.jpg

# fake = 0
# real = 1

# happy = 0
# sad = 1
# surprise = 2
# disgust = 3
# contempt = 4
# angry = 5

# Rename file should look like this:
# 1_5_H2N2A.MP4age103.jpg

from tqdm.auto import tqdm



def rename_real_fake(dir_image_paths):
    # for file in dir_image_paths:
    print("Rename the real fake started")
    for file in tqdm(dir_image_paths):
        # print("File name: ", file)
        if file.split("_")[0] == "real":
            real_fake_name = file.replace("real", "1")
            os.rename(os.path.join(dir, file), os.path.join(dir, real_fake_name))

        elif file.split("_")[0] == "fake":
            real_fake_name = file.replace("fake", "0")
            os.rename(os.path.join(dir, file), os.path.join(dir, real_fake_name))
    print("Rename the real fake finished")


# Rename the emotions:
def rename_emotions(dir_image_paths):
    print("Rename the emotions started")
    for file in tqdm(dir_image_paths):
        if file.split("_")[1] == "happy":
            emotion_name = file.replace("happy", "0")
            os.rename(os.path.join(dir, file), os.path.join(dir, emotion_name))

        elif file.split("_")[1] == "sad":
            emotion_name = file.replace("sad", "1")
            os.rename(os.path.join(dir, file), os.path.join(dir, emotion_name))

        elif file.split("_")[1] == "surprise":
            emotion_name = file.replace("surprise", "2")
            os.rename(os.path.join(dir, file), os.path.join(dir, emotion_name))

        elif file.split("_")[1] == "disgust":
            emotion_name = file.replace("disgust", "3")
            os.rename(os.path.join(dir, file), os.path.join(dir, emotion_name))

        elif file.split("_")[1] == "contempt":
            emotion_name = file.replace("contempt", "4")
            os.rename(os.path.join(dir, file), os.path.join(dir, emotion_name))

        elif file.split("_")[1] == "angry":
            emotion_name = file.replace("angry", "5")
            os.rename(os.path.join(dir, file), os.path.join(dir, emotion_name))
    print("Rename the emotions finished")




dir = 'data_mtl/train'

dir_image_paths = os.listdir(dir)

# rename_real_fake(dir_image_paths)

rename_emotions(dir_image_paths)
