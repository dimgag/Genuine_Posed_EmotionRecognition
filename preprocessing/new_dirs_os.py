import os 
import shutil
 

data_dir = "data/SASE-FE/FakeTrue_DB"
persons = os.listdir(data_dir)

emotions = ['real_surprise','real_angry','real_happy','real_sad','real_disgust','real_contempt',
                'fake_surprise','fake_angry','fake_happy','fake_sad','fake_disgust','fake_contempt']



# List files dirs
# print(os.listdir(data_dir))

# Example:
# r data/SASE-FE/FakeTrue_DB/Pavel/N2S.MP4
# r_subdir data/SASE-FE/FakeTrue_DB/Pavel
# r_file N2S.MP4

# Copy first 40 persons to train_prep folder
for person in persons:
    if ".DS_Store" in person:
        persons.remove(".DS_Store")
    for person in persons[:40]:
        # copy persons folder to train_prep
        if not os.path.exists("data/train_prep/" + person):
            shutil.copytree(data_dir + "/" + person, "data/train_prep/" + person)
    for person in persons[40:]:
        # copy persons folder to test_prep
        if not os.path.exists("data/test_prep/" + person):
            shutil.copytree(data_dir + "/" + person, "data/test_prep/" + person)
print("Copied files")




train_prep = "data/train_prep"
test_prep = "data/test_prep"

# Now I have splitted the participants in two folders with 40 persons in train (80%) and 10 persons in test (20%)
# Those folders have the original videos of the participants
# 1. I want to extract frames from those videos and put them in emotions folders

# for emotion in emotions:
#     if not os.path.exists("data/train_prep/" + emotion):
#         os.mkdir("data/train_prep/" + emotion)
#     if not os.path.exists("data/test_prep/" + emotion):
#         os.mkdir("data/test_prep/" + emotion)

# Extract video frames and put them in emotions folders

from preprocessing_utils import get_files_paths, video2frames

# Train_prep folder
r, r_subdir, r_file = get_files_paths(train_prep)

[os.mkdir('data/train_prep/' + emotion) for emotion in emotions]

# Extract frames from videos in train_prep folder
for video, dirname, file in zip(r, r_subdir, r_file):
    video2frames(video, dirname, file, 'train_prep')

# Delete persons folders in train_prep
for person in persons[:40]:
    shutil.rmtree('data/train_prep/' + person)



# Test_prep folder
r, r_subdir, r_file = get_files_paths(test_prep)

[os.mkdir('data/test_prep/' + emotion) for emotion in emotions]

# Extract frames from videos in test_prep folder
for video, dirname, file in zip(r, r_subdir, r_file):
    video2frames(video, dirname, file, 'test_prep')

# Delete persons folders in train_prep
for person in persons[40:]:
    shutil.rmtree('data/test_prep/' + person)


print("EXTRACTING FRAMES IS DONE")

# 2. Now I want to crop the frames and put them in emotions folders

