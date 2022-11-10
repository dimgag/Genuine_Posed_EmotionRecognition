import numpy as np # Numerical library for operations
import math # Mathematical library 
import glob # Iteration between paths
from shutil import copyfile # Method to copy files
import pandas as pd # Pandas for best data structures
import cv2 # OpenCV V3.3.0
import random # Random generations
%matplotlib inline
#pd.options.display.max_colwidth = 100
import matplotlib.pyplot as plt #Matplot object to plot
from sklearn.svm import SVC # SVM sklearn implementation
from sklearn.ensemble import RandomForestClassifier # RandomForest sklearn implementation
from sklearn import metrics # Metrics object for calculations
from sklearn.feature_selection import RFECV # Wrapper that automatically decreases and choose best model
from sklearn.model_selection import train_test_split # Method to split into train/test data
from sklearn.model_selection import cross_val_score # Method for cross-validation and score together
from sklearn.model_selection import GridSearchCV # Method with cross-validation and grid search parameters
import pickle #lib to save the model trained into file

from keras.utils import np_utils # util of Keras
from keras.models import Sequential, Model # Types of model to be used
from keras.layers import Dense, Dropout, Activation, Flatten # Dense layers of NN
from keras.layers import Conv2D, MaxPooling2D # Convolution layers related of NN
import keras.backend as K # Changing to channels_first input layout
K.set_image_data_format('channels_first')
K.set_floatx('float32')
print('Backend:        {}'.format(K.backend()))
print('Data format:    {}'.format(K.image_data_format()))
print(os.listdir("Datasets/CK+"))