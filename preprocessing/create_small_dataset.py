# Get a small version of the origianl dataset for local use
# Author: Dimitrios Gagatsis
import shutil
import os


# Load Images
# Data Directory
data_dir = 'data/SASE-FE/frames'
real = 'data/SASE-FE/frames/real'
fake = 'data/SASE-FE/frames/fake'
# List directories in real and fake
real_emotions = os.listdir(real)
fake_emotions = os.listdir(fake)

real_emotions.remove('.DS_Store')
fake_emotions.remove('.DS_Store')

# Create a new directory
os.mkdir("data/SASE-FE/frames_small")

# Create subdirectories
os.mkdir("data/SASE-FE/frames_small/real")
os.mkdir("data/SASE-FE/frames_small/fake")

# Copy 100 images from each emotion to the new directory
for emotion in real_emotions:
    os.mkdir("data/SASE-FE/frames_small/real/" + emotion)
    source = real + "/" + emotion
    dest = "data/SASE-FE/frames_small/real/" + emotion
    files = os.listdir(source)
    for i in range(100):
        shutil.copy(source + "/" + files[i], dest)

for emotion in fake_emotions:
    os.mkdir("data/SASE-FE/frames_small/fake/" + emotion)
    source = fake + "/" + emotion
    dest = "data/SASE-FE/frames_small/fake/" + emotion
    files = os.listdir(source)
    for i in range(100):
        shutil.copy(source + "/" + files[i], dest)

# Load the small dataset
data_dir = 'data/SASE-FE/frames_small'
real = 'data/SASE-FE/frames_small/real'
fake = 'data/SASE-FE/frames_small/fake'

