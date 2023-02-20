# Prepare data_temporal folder

import os
# !mkdir data_temporal
# don't forget to add /data_temporal/* to .gitignore file.
data_dir = 'data_temporal'

# Create train and test folders
os.makedirs(os.path.join(data_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'test'), exist_ok=True)

# copy folder FakeTrue_DB to data_temporal
# !mkdir data_temporal
# !cd data_temporal 
# !mkdir 
# !cp -r data/SASE-FE/FakeTrue_DB data_temporal


# Data Structure: 
# data_temporal
#     FakeTrue_DB  # this is the folder with the videos
#               / Ahmed 
#                       / D2N2Sur.MP4
#                       / H2N2A.MP4
#                       / H2N2C.MP4
#                       / H2N2D.MP4
#                       / H2N2S.MP4
#                       / N2A.MP4
#                       / N2C.MP4
#                       / N2D.MP4
#                       / N2H.MP4
#                       / N2S.MP4
#                       / N2Sur.MP4
#                       / S2N2H.MP4

#               / yirri

