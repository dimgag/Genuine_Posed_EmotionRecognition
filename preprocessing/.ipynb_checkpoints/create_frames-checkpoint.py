# This file generates frames videos and splits them into 12 labels.
# 'real_surprise','real_angry','real_happy','real_sad','real_disgust','real_contempt',
# 'fake_surprise','fake_angry','fake_happy','fake_sad','fake_disgust','fake_contempt'
# Author: Dimitrios Gagatsis

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
    '''Function to remove .DS_Store files
    and get the file paths'''
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
                for file in files:
                    if file == '.DS_Store':
                        os.remove(os.path.join(subdir, file))
                        print("Removed .DS_Store file from sub-directory")
                    r.append(os.path.join(subdir, file))
                    r_subdir.append(subdir)
                    r_file.append(file)
    return r, r_subdir, r_file


def video2frames(dir, dirname, file):
    '''Function to extract frames from videos'''
    vidcap = cv2.VideoCapture(dir)
    length_of_video = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, image = vidcap.read()

    new_dir = dir.replace('patches', 'frames3D').split('.MP4')[0]
    print("New directory for the frames: ", new_dir)
    print(os.path.basename(new_dir))

    count = 0
    frame_counter = 0
    vid = []
    while success:
        success, image = vidcap.read()

        if frame_counter > 16:
            frame_counter = 0
            print(file)
            vid = []

        if count > int(length_of_video*.6) and count < int(length_of_video*.9):
            vid.append(dirname)
            if file == 'N2SUR.MP4':            
                print(file, 'real', 'surprise')
                cv2.imwrite(os.path.join('data/frames/real_surprise', file + dirname.split('/')[-1] + dirname.split('/')[-1] +"{:02d}.jpg".format(count) ), image)
            if file == 'N2A.MP4':
                print(file, 'real', 'angry')
                cv2.imwrite(os.path.join('data/frames/real_angry', file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'N2C.MP4':
                print(file, 'real', 'contempt')
                cv2.imwrite(os.path.join('data/frames/real_contempt', file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'N2D.MP4':
                print(file, 'real', 'disgust')
                cv2.imwrite(os.path.join('data/frames/real_disgust', file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'N2S.MP4':
                print(file, 'real', 'sad')
                cv2.imwrite(os.path.join('data/frames/real_sad', file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'N2H.MP4':
                print(file, 'real', 'happy')
                cv2.imwrite(os.path.join('data/frames/real_happy', file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'D2N2SUR.MP4':
                print(file, 'fake', 'surprise')
                cv2.imwrite(os.path.join('data/frames/fake_surprise', file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'H2N2A.MP4':
                print(file, 'fake', 'angry')
                cv2.imwrite(os.path.join('data/frames/fake_angry', file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'H2N2C.MP4':
                print(file, 'fake', 'contempt')
                cv2.imwrite(os.path.join('data/frames/fake_contempt', file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'H2N2D.MP4':
                print(file, 'fake', 'disgust')
                cv2.imwrite(os.path.join('data/frames/fake_disgust', file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'H2N2S.MP4':
                print(file, 'fake', 'sad')
                cv2.imwrite(os.path.join('data/frames/fake_sad', file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'S2N2H.MP4':
                print(file, 'fake', 'happy')
                cv2.imwrite(os.path.join('data/frames/fake_happy', file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
        if cv2.waitKey(10) == 27:
            break

        count += 1
        frame_counter += 1





if __name__ == "__main__":
    # data directory
    # data_dir = "data/SASE-FE/FakeTrue_DB"
    # DSRI Directory with the data
    data_dir = "data/FakeTrue_DB"
    
    # Create new directories for the frames
    create_dir('data/frames') #Edit the name here later
    frames_dir = 'data/frames'
    # Create subdirs for the emotions
    emotions = ['real_surprise','real_angry','real_happy','real_sad','real_disgust','real_contempt',
                'fake_surprise','fake_angry','fake_happy','fake_sad','fake_disgust','fake_contempt']

    [create_dir('data/frames/' + emotion) for emotion in emotions]
    # Get the files paths
    r, r_subdir, r_file = get_files_paths(data_dir)

    # Capitalize all video names
    r_file = [file.upper() for file in r_file]

    # Extract frames from videos
    for video, dirname, file in zip(r, r_subdir, r_file):
        video2frames(video, dirname, file)

    print("Frames extracted successfully!")

    # Count the number of images in each folder
    for emotion in emotions:
        print(emotion, len(os.listdir(os.path.join(frames_dir, emotion))))