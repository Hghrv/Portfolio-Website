# Importing Keras and other necessary libraries for the Python API of Tensorflow
import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
import h5py
import sklearn
import sklearn.datasets
import scipy

from PIL import Image
from scipy import ndimage


from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras import regularizers

np.random.seed(7)


# %matplotlib inline

# Loading the training dataset
def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    f.seek(0)
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

training_data, validation_data, test_data = load_data()

training_data

# Setting the one_hot encoding
def one_hot(j):
    # input is the target dataset of shape (1, m) where m is the number of data points
    # returns a 2 dimensional array of shape (10, m) where each target value is converted to a one hot encoding
    # Look at the next block of code for a better understanding of one hot encoding
    n = j.shape[0]
    new_array = np.zeros((10, n))
    index = 0
    for res in j:
        new_array[res][index] = 1.0
        index = index + 1
    return new_array

data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
one_hot(data)


