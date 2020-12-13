import scipy.io as spio
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

train_path = '/Users/saif/Desktop/University/Year 4/CI/Final/datasets/training.mat'

tmat = spio.loadmat(train_path, squeeze_me=True)
cl = tmat['Class']
cl = np.subtract(cl, 1)

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y.T

C = np.max(cl) + 1
y = convert_to_one_hot(cl, C)

X = np.load('data.npy')
print(X.shape)
print(y.shape)


