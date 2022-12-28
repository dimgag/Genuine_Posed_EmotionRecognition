# This file generates frames videos and splits them into 12 labels.
# The labels are: 
# 'real_surprise','real_angry','real_happy','real_sad','real_disgust','real_contempt',
# 'fake_surprise','fake_angry','fake_happy','fake_sad','fake_disgust','fake_contempt'

import os
import cv2


def create_dir(dir):
    '''Create directory function'''
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Created directory: ", dir)
    else:
        print("Directory already exists")


def get_files_paths(dir):
    if '.DS_Store' in os.listdir(dir):
        os.remove(os.path.join(dir, '.DS_Store'))
        print("Remove .DS_Store file from main directory")

    r = []
    r_subdir = []
    r_file = []

    subdirs = [x[0] for x in os.walk(dir)]

    for subdir in subdirs:
        if '.DS_Store' in subdir:
            os.remove(subdir)
            print("Remove .DS_store file from sub-directory", subdir)
        else:
            files = os.walk(subdir).__next__()[2]
            if (len(files) > 0):
                # t.b.c.





















if __name__ == "__main__":
    # data directory
    data_dir = "data/SASE-FE/FakeTrue_DB"
    # Create new directories for the frames
    create_dir('data/frames_new') #Edit the name here later
    frames_dir = 'data/frames_new'
    # Create subdirs for the emotions
    emotions = ['real_surprise','real_angry','real_happy','real_sad','real_disgust','real_contempt',
                'fake_surprise','fake_angry','fake_happy','fake_sad','fake_disgust','fake_contempt']

    [create_dir('data/frames/' + emotion) for emotion in emotions]
    # Get the files paths
    r, r_subdirs, r_files = get_files_paths(data_dir)

    # Capitalize all video names
    r_files = [file.upper() for file in r_files]

    # Extract frames from videos
    # t.b.c.  :)