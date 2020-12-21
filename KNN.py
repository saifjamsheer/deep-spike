import scipy.io as spio
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import cluster as cl
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Examining the shapes of the various datasets
print("number of training examples = {}".format(X_train.shape[0]))
print("number of test examples = {}".format(X_test.shape[0]))
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

n_x = X_train.shape[1]
n_y = y_train.shape[1]

def pca_components(train, test):
    """
    Calculate ideal number of components for
    PCA dimensionality reduction

    train: train dataset (x values)
    test: test dataset (x values)

    """
    pca = PCA()
    train = pca.fit_transform(train)
    test = pca.transform(test)
    t = sum(pca.explained_variance_)
    n_c = 0 
    variance = 0
    while variance/t < 0.95:
        variance += pca.explained_variance_[n_c]
        n_c += 1
    
    return n_c

n_c = pca_components(X_train, X_test)

pca = PCA(n_components=n_c)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

model = KNeighborsClassifier(n_neighbors=5, p=2) 
model.fit(X_train, y_train)

predict = model.predict(X_test)
predict = np.rint(predict)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_predict = np.rint(train_predict)
test_predict = np.rint(test_predict)

train_accuracy = accuracy_score(y_train, train_predict)
test_accuracy = accuracy_score(y_test, test_predict)
print("Train Accuracy: {val}".format(val=train_accuracy))
print("Test Accuracy: {val}".format(val=test_accuracy))
print("..............................................")