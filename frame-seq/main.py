import os
import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing import image
import numpy as np
from keras.utils import np_utils
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm

# Data dir:
data_train = 'data_temporal/train_root'
data_val = 'data_temporal/val_root'

def get_data_csvs(data_folder, filename):
    ''' Extracts the video paths and labels from the dataset and saves them in a csv file'''
    # Create a dataframe with video path and label
    df = pd.DataFrame(columns=['Video_path', 'label'])
    # avoid .DS_Store

    def remove_ds_store(directory):
        for dirpath, dirnames, filenames in os.walk(directory):
            # Remove .DS_Store files in the current directory
            if ".DS_Store" in filenames:
                os.remove(os.path.join(dirpath, ".DS_Store"))
                print(f"Removed .DS_Store file from {dirpath}")

            # Recursively remove .DS_Store files in subdirectories
            for dirname in dirnames:
                subdir = os.path.join(dirpath, dirname)
                remove_ds_store(subdir)

    remove_ds_store(data_folder)

    for folder in os.listdir(data_folder):
        for video in os.listdir(os.path.join(data_folder, folder)):
            # print(folder)
            df = df.append({'Video_path':os.path.join(data_folder, folder, video), 'label':folder}, ignore_index=True)

    # save the dataframe as a csv file
    df.to_csv('frame-seq/' + filename + '.csv', index=False)

# get_data_csvs(data_train, 'traincsv')
# get_data_csvs(data_val, 'valcsv')

def create_csv_files():
    data_train = 'data_temporal/train_root'
    data_val = 'data_temporal/val_root'
    get_data_csvs(data_train, 'traincsv')
    get_data_csvs(data_val, 'valcsv')




def get_val_frames():
    val = pd.read_csv('frame-seq/valcsv.csv')
    ''' Extracts frames from a video and returns a list of frames'''
    for i in tqdm(range(val.shape[0])):
        count = 0
        videoFile = val['Video_path']
        cap = cv2.VideoCapture(videoFile[i])  # capturing the video from the given path
        frameRate = cap.get(5)  # frame rate
        x = 1
        while(cap.isOpened()):
            frameId = cap.get(1)
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frameId % math.floor(frameRate) == 0):
                filename = "frame%d.jpg" % count;
                count += 1
                cv2.imwrite('frame-seq/val_1' + "/" + val['label'][i] + "/" + filename, frame)
        cap.release()


if __name__ == "__main__":
    train = pd.read_csv('frame-seq/traincsv.csv')
    get_val_frames()

# WHERE ARE THE FRAMES GOING?