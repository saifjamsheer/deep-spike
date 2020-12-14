import scipy.io as spio
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input
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
# index = 6; plt.plot(X[index]); plt.show()

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Examing the shapes of the various datasets
print("number of training examples = {}".format(X_train.shape[0]))
print("number of test examples = {}".format(X_test.shape[0]))
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

model = Sequential()
model.add(Dense(7, input_dim=75, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=16)

train_predict = model.predict(X_train, batch_size=16)
test_predict = model.predict(X_test, batch_size=16)
train_predict = np.rint(train_predict)
test_predict = np.rint(test_predict)

print(np.where(test_predict==1)[1])
print(np.where(y_test==1)[1])

train_accuracy = accuracy_score(y_train, train_predict)
test_accuracy = accuracy_score(y_test, test_predict)
print("Train Accuracy: {val}".format(val=train_accuracy))
print("Test Accuracy: {val}".format(val=test_accuracy))
print("..............................................")

