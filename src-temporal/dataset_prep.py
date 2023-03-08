# File to prepare the dataset in the format of TimeSformer
import os
import shutil
import random
import pandas as pd
import subprocess

# Run this "rm -rf `find -type d -name .ipynb_checkpoints`" in terminal to avoid .ipynb_checkpoints

subprocess.run(["rm", "-rf", "`find", "-type", "d", "-name", ".ipynb_checkpoints`"])


def rename_filenames(directory):
    for dirpath, dirnames, filenames in os.walk(directory):
        # Rename directories in the current directory
        for filename in filenames:
            old_filename = os.path.join(dirpath, filename)
            new_filename = os.path.join(dirpath, filename.upper())
            os.rename(old_filename, new_filename)

def remove_ds_store(directory):
    for dirpath, dirnames, filenames in os.walk(directory):
        # Remove .DS_Store files in the current directory
        if ".DS_Store" in filenames:
            os.remove(os.path.join(dirpath, ".DS_Store"))
            print(f"Removed .DS_Store file from {dirpath}")
            
        if ".ipynb_checkpoints" in filenames:
            os.remove(os.path.join(dirpath, ".ipynb_checkpoints"))
            print(f"Removed .ipynb_checkpoints file from {dirpath}")

        # Recursively remove .DS_Store files in subdirectories
        for dirname in dirnames:
            subdir = os.path.join(dirpath, dirname)
            remove_ds_store(subdir)


def convert_dataset(input_dir, output_dir, train_val_split):
    # Create output directories
    train_root = os.path.join(output_dir, 'train_root')
    val_root = os.path.join(output_dir, 'val_root')
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)

    # Get the list of participants
    participants = os.listdir(input_dir)

    # Shuffle participants and split into train and validation sets
    random.shuffle(participants)
    split_idx = int(len(participants) * train_val_split)
    train_participants = participants[:split_idx]
    val_participants = participants[split_idx:]

    # Move videos to corresponding directories
    for participant in train_participants:
        participant_dir = os.path.join(input_dir, participant)
        for video in os.listdir(participant_dir):
            src_path = os.path.join(participant_dir, video)
            label = os.path.splitext(video)[0]
            dst_dir = os.path.join(train_root, label)
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, f"{label}_{participant}.mp4")
            shutil.copy(src_path, dst_path)

    for participant in val_participants:
        participant_dir = os.path.join(input_dir, participant)
        for video in os.listdir(participant_dir):
            src_path = os.path.join(participant_dir, video)
            label = os.path.splitext(video)[0]
            dst_dir = os.path.join(val_root, label)
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, f"{label}_{participant}.mp4")
            shutil.copy(src_path, dst_path)

import os

def rename_folders(directory):
    mapping = {
        "D2N2SUR": "fake_surprise",
        "H2N2A": "fake_angry",
        "H2N2C": "fake_contempt",
        "H2N2D": "fake_disgust",
        "H2N2S": "fake_sad",
        "S2N2H": "fake_happy",
        "N2A": "real_angry",
        "N2C": "real_contempt",
        "N2D": "real_disgust",
        "N2H": "real_happy",
        "N2S": "real_sad",
        "N2SUR": "real_surprise"
    }

    for dirpath, dirnames, filenames in os.walk(directory):
        # Rename directories in the current directory
        for dirname in dirnames:
            old_dirname = os.path.join(dirpath, dirname)
            new_dirname = mapping.get(dirname, dirname)
            if new_dirname != dirname:
                new_dirname = os.path.join(dirpath, new_dirname)
                os.rename(old_dirname, new_dirname)

def rename_filenames(directory):
    for dirpath, dirnames, filenames in os.walk(directory):
        # remove spaces in filenames
        for filename in filenames:
            old_filename = os.path.join(dirpath, filename)
            new_filename = os.path.join(dirpath, filename.replace(" ", ""))
            os.rename(old_filename, new_filename)


def get_data_csvs(data_folder, filename):
    ''' Extracts the video paths and labels from the dataset and saves them in a csv file'''
    # Create a dataframe with video path and label
    df = pd.DataFrame(columns=['Video_path', 'Label'])
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
            df = df.append({'Video_path':os.path.join(data_folder, folder, video), 'Label':folder}, ignore_index=True)
            # Add numbers to the labels to make them integers
            mapping = {'fake_surprise':0,
                        'fake_angry':1,
                        'fake_contempt':2,
                        'fake_disgust':3,
                        'fake_sad':4,
                        'fake_happy':5,
                        'real_angry':6,
                        'real_contempt':7,
                        'real_disgust':8,
                        'real_happy':9,
                        'real_sad':10,
                        'real_surprise':11}
            # rename the labels
            if folder in mapping:
                df['Label'] = df['Label'].replace(folder, mapping[folder])

            
            



                       

    # save the dataframe as a csv file
    df.to_csv('data_temporal/' + filename + '.csv', sep=' ', index=False)






if __name__ == '__main__':    
    # input_dir = 'data_temporal/FakeTrue_DB'
    # output_dir = 'data_temporal'
    # train_val_split = 0.8
    # rename_filenames(input_dir)
    # remove_ds_store(input_dir)
    # convert_dataset(input_dir, output_dir, train_val_split)
    # rename_folders('data_temporal/train_root')
    # rename_folders('data_temporal/val_root')
    rename_filenames('data_temporal/train_resized')
    rename_filenames('data_temporal/val_resized')
    get_data_csvs('data_temporal/train_resized', 'train')
    get_data_csvs('data_temporal/val_resized', 'val')



    
    