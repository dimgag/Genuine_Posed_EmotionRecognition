# Data functions for tensorflow models
# Author: Dimitrios Gagatsis
import os

from keras.preprocessing.image import ImageDataGenerator



image_size = 224
batch_size = 32
num_workers = 4

## Data directoris
# Local Paths
train_dir = "data/train"
test_dir = "data/test"

train_images = os.listdir(train_dir)
test_images = os.listdir(test_dir)


# Remove .DS_Store files
if '.DS_Store' in train_images:
    train_images.remove('.DS_Store')
if '.DS_Store' in test_images:
    test_images.remove('.DS_Store')


# # # # # # # # # # # # # # # # # # # # # # 
# Data Augmentation
# Training Data Transforms
def get_train_transform(image_size):
    train_transform = ImageDataGenerator(
        rescale=1./255
    )
    return train_transform

# Test Data Transforms
def get_test_transform(image_size):
    test_transform = ImageDataGenerator(
        rescale=1./255
    )
    return test_transform

def get_datasets():
    """
    Function to prepare the Datasets.
    Returns the training and Test datasets along 
    with the class names.
    """
    dataset_train = get_train_transform(image_size).flow_from_directory(
        train_dir, 
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical'
    )
    dataset_test = get_test_transform(image_size).flow_from_directory(
        test_dir, 
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical'
    )
    return dataset_train, dataset_test

def get_class_names():
    """
    Function to get the class names.
    Returns the class names.
    """
    class_names = os.listdir(train_dir)
    return class_names

def get_num_classes():
    """
    Function to get the number of classes.
    Returns the number of classes.
    """
    num_classes = len(os.listdir(train_dir))
    return num_classes



# train_loader, test_loader = get_datasets()
# class_names = get_class_names()
# num_classes = get_num_classes()
