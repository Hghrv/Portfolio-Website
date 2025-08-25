# Importing Keras and other necessary libraries for the Python API of Tensorflow
import tensorflow as tf
import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Printing datasets details
print("The feature dataset is:" + str(training_data[0]))
print("The target dataset is:" + str(training_data[1]))
print("The number of examples in the training dataset is:" + str(len(training_data[0])))
print("The number of points in a single input is:" + str(len(training_data[0][1])))

# Setting the one_hot encoding for the target value
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

# Defining the data_wrapper function
def data_wrapper():
    tr_d, va_d, te_d = load_data()
    
    training_inputs = np.array(tr_d[0][:]).T
    training_results = np.array(tr_d[1][:])
    train_set_y = one_hot(training_results)
    
    validation_inputs = np.array(va_d[0][:]).T
    validation_results = np.array(va_d[1][:])
    validation_set_y = one_hot(validation_results)
    
    test_inputs = np.array(te_d[0][:]).T
    test_results = np.array(te_d[1][:])
    test_set_y = one_hot(test_results)
    
    return (training_inputs, train_set_y, validation_inputs, validation_set_y)

# Calling the data_wrapper() function and assigning the output to local variables
train_set_x, train_set_y, test_set_x, test_set_y = data_wrapper()

# Transposing the sets
train_set_x = train_set_x.T
train_set_y = train_set_y.T
test_set_x = test_set_x.T
test_set_y = test_set_y.T

# Checking that the sets are in the desired shape
print ("train_set_x shape: " + str(train_set_x.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

# Visualising the dataset by index to check correct labelling
index  = 1000
k = train_set_x[index,:]
k = k.reshape((28, 28))
plt.title('Label is {label}'.format(label= training_data[1][index]))
plt.imshow(k, cmap='gray')

# Creating first instance of sequential neural model and adding density (layers, activation function, regulariser)
# First instance without layers
nn_model = Sequential()

# Adding a Dropout for the first layer at 0.3% of neurons in Bayesian dropout for each iteration in order to generate a sparse weight matrix after cumulative input
nn_model.add(Dropout(0.3))

# Initialising first hidden layer with 35 neurons, 28x28 = 784 components in the input vectors and 'relu' activation function
nn_model.add(Dense(35, input_dim=784, activation='relu'))

# Regularising the interconnected neural network
nn_model.add(Dense(21, activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))

# Suggestions for network initialization
# w=np.random.randn(layer_size[l],layer_size[l-1])*0.01
# w=np.random.randn(layer_size[l],layer_size[l-1])*np.sqrt(2/layer_size[l-1])
# w=np.random.randn(layer_size[l],layer_size[l-1])*np.sqrt(2/(layer_size[l-1]+layer_size{l]))
# w=np.random.randn(layer_size[l],layer_size[l-1])*(np.sqrt(6/(layer_size[l-1]))+(np.sqrt(6/layer_size{l])))

# Setting the last softmax layer with 10 classes 
nn_model.add(Dense(10, activation='softmax'))

# Compiling the model with the crossentropic loss function
nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fitting the model with a minibatch of size 10 and 10 epochs
nn_model.fit(train_set_x, train_set_y, epochs=10, batch_size=10)

# Evaluating the model's scores and printing the accuracy in the training dataset
scores_train = nn_model.evaluate(train_set_x, train_set_y)
print("\n%s: %.2f%%" % (nn_model.metrics_names[1], scores_train[1]*100))

# Setting the predictions on the test dataset
predictions = nn_model.predict(test_set_x)
predictions = np.argmax(predictions, axis = 1)
predictions

# Setting scores and printing the accuracy in the test dataset
scores_test = nn_model.evaluate(test_set_x, test_set_y)
print("\n%s: %.2f%%" % (nn_model.metrics_names[1], scores_test[1]*100))

# Visualising different test cases for assessment against validation data
index  = 9997
k = test_set_x[index, :]
k = k.reshape((28, 28))
plt.title('Label is {label}'.format(label=(predictions[index], np.argmax(test_set_y, axis = 1)[index])))
plt.imshow(k, cmap='gray')