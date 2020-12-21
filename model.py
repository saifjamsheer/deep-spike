import scipy.io as spio
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score

train_path = '/Users/saif/Desktop/University/Year 4/CI/Final/datasets/training.mat'

# Loading dataset
tmat = spio.loadmat(train_path, squeeze_me=True)

# Converting neuron values from 1-4 to 0=3
cl = tmat['Class']
cl = np.subtract(cl, 1)

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y.T

# One-hot encoding of classes
C = np.max(cl) + 1
y = convert_to_one_hot(cl, C)

# Loading inputs
X = np.load('data.npy')

# Example of a peak
# index = 3074; plt.plot(X[index]); plt.show()

# Splitting dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Randomly shuffling datasets in unison 
p = np.random.permutation(len(X))
X, y = X[p], y[p]

# Splitting dataset
# pylint: disable=unbalanced-tuple-unpacking
X_train, X_val, X_test = np.split(X, [int(.6*len(X)), int(.8*len(X))])
# pylint: disable=unbalanced-tuple-unpacking
y_train, y_val, y_test = np.split(y, [int(.6*len(y)), int(.8*len(y))])

# Examining the shapes of the various datasets
print("number of training examples = {}".format(X_train.shape[0]))
print("number of validation examples = {}".format(X_val.shape[0]))
print("number of test examples = {}".format(X_test.shape[0]))
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_val shape: {}".format(X_val.shape))
print("y_val shape: {}".format(y_val.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

n_x = X_train.shape[1]
n_y = y_train.shape[1]

def build(n_x, n_y, h_layers, learning_rate=0.001):

    model = Sequential()
    model.add(Input(shape=n_x))

    for nodes in h_layers:
        model.add(Dense(nodes, activation='relu'))

    optimizer = Adam(learning_rate=learning_rate)

    model.add(Dense(n_y, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

# Building the model
model = build(n_x, n_y, [7, 7])

# Fitting the model to the training data
history = model.fit(x=X_train, 
                    y=y_train, 
                    epochs=50, 
                    batch_size=16, 
                    validation_data=(X_val, y_val))

test_predict = model.predict(X_test, batch_size=16)
test_predict = np.rint(test_predict)

# Validation and testing accuracy
val_accuracy = history.history['val_accuracy'][-1]
test_accuracy = accuracy_score(y_test, test_predict)
print("Validation Accuracy: {val}".format(val=val_accuracy))
print("Test Accuracy: {val}".format(val=test_accuracy))
print("..............................................")

def convert_from_one_hot(Y):
    return np.where(Y==1)[1]