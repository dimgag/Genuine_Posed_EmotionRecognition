# Split the data into train and test directories 80/20
# Move images from cropped_faces to train and test directories
# Author: Dimitrios Gagatsis
import os 

# Get the cropped_faces directory
faces = '../data/cropped_faces'
faces = os.listdir(faces)

# If .DS_Store is in the directory, remove it
if '.DS_Store' in faces:
    faces.remove('.DS_Store')


for emotion in faces:
    if '.DS_Store' in emotion:
        emotion.remove('.DS_Store')


# Create the train and test directories
if not os.path.exists('../data/train'):
    os.mkdir('../data/train')

if not os.path.exists('../data/test'):
    os.mkdir('../data/test')

# Create the emotion directories in the train and test directories
for emotion in faces:
    if not os.path.exists('../data/train/' + emotion):
        os.mkdir('../data/train/' + emotion)
    if not os.path.exists('../data/test/' + emotion):
        os.mkdir('../data/test/' + emotion)

# Move the images into the train and test directories 80/20
for emotion in faces:
    images = os.listdir('../data/cropped_faces/' + emotion)
    for image in images:
        if '.DS_Store' in image:
            images.remove('.DS_Store')
    for image in images[:int(len(images)*0.8)]:
        os.rename('../data/cropped_faces/' + emotion + '/' + image, '../data/train/' + emotion + '/' + image)
    for image in images[int(len(images)*0.8):]:
        os.rename('../data/cropped_faces/' + emotion + '/' + image, '../data/test/' + emotion + '/' + image)


# How many images in train and test directories
train = 0
test = 0
for emotion in faces:
    train += len(os.listdir('../data/train/' + emotion))
    test += len(os.listdir('../data/test/' + emotion))

print('Train images: ', train)
print('Test images: ', test)
