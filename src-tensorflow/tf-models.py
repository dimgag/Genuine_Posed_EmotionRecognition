import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD

# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#     raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))

import os
import numpy as np

# Get VGG16 model

VGG16 = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))


# Freeze the layers except the last 4 layers
for layer in VGG16.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in VGG16.layers:
    print(layer, layer.trainable)

# Create the model
model = Sequential()

# Add the vgg convolutional base model
model.add(VGG16)

# Add new layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()

