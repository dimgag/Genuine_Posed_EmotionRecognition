import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from data import get_datasets

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    train_loader, test_loader = get_datasets()

    
