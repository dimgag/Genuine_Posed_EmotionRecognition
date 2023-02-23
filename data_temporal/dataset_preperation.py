# File to prepare the dataset in the format of TimeSformer
import os
import shutil
import random

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
        "D2N2Sur": "fake_surprise",
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
        "N2Sur": "real_surprise"
    }

    for dirpath, dirnames, filenames in os.walk(directory):
        # Rename directories in the current directory
        for dirname in dirnames:
            old_dirname = os.path.join(dirpath, dirname)
            new_dirname = mapping.get(dirname, dirname)
            if new_dirname != dirname:
                new_dirname = os.path.join(dirpath, new_dirname)
                os.rename(old_dirname, new_dirname)



input_dir = 'data_temporal/FakeTrue_DB'
output_dir = 'data_temporal'
train_val_split = 0.8

remove_ds_store(input_dir)
convert_dataset(input_dir, output_dir, train_val_split)
rename_folders('data_temporal/train_root')
rename_folders('data_temporal/val_root')