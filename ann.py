import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

def filter(signal, cutoff, type, fs=25000, order=2):
    """
    Attenuates noise from the inputted signal and returns the filtered
    signal.

    INPUTS
    * signal:   raw time domain recordings (numpy array)
    * cutoff:   cut-off frequency in Hz
    * type:     type of filter (e.g. 'low', 'high', 'bandpass')
    * fs:       sampling frequency of signal (25000 Hz)
    * order:    order of filter (default = 2)

    OUTPUTS
    * filtered: filtered signal
    """
    # Reduce noise in signal based on cut-off frequency and filter type
    sos = butter(order, cutoff, btype=type, analog=False, output='sos', fs=fs)
    filtered = sosfiltfilt(sos, signal)

    return filtered

def detect(signal):
    """
    Detects the peaks within the inputted signal and returns the positions
    of these peaks.

    INPUTS
    * signal:   raw time domain recordings (numpy array)

    OUTPUTS
    * peaks:    vector representing positions of peaks within the window
    """
    # Calculate median absolute deviation of the points in the signal
    mad = np.median(np.absolute(signal)/0.6745) 
    # Set a threshold for peak detection
    threshold = 5 * mad
    # Use find_peaks to find positions of peaks in the signal
    peaks, _ = find_peaks(signal, height=threshold)

    return peaks

def extract(signal, peaks, window, idx, neurons):
    """
    Extracts data points distributed around spike peaks based on the
    window size.

    INPUTS
    * signal:   raw time domain recordings (numpy array)
    * peaks:    vector representing positions of peaks within the signal
    * window:   list with the number of points to the left and right
                of each peak that should be extracted
    * idx:      Index vector from training.mat
    * neurons:  Class vector from training.mat

    OUTPUTS
    * spikes:   Array of extracted sample points
    * classes:  Array of corresponding neuron types
    """
    spikes = [] # Initialize array of sample points
    classes = [] # Initialize array of neuron types
    duplicates = [] # Initialize duplicates array

    # Iterate through the positions of peaks
    for peak_idx in peaks:
        # Find the index value corresponding to each peak
        i = idx[idx < peak_idx].max()
        # Prevent duplicates from being added
        if i in duplicates: continue
        duplicates.append(i)
        # Find the position of i within the Index vector
        n = np.where(idx==i)[0][0]
        # Extract data points to the left and right of a peak
        values = signal[peak_idx-window[0]:peak_idx+window[1]]

        spikes.append(values)
        classes.append(neurons[n])

    return spikes, classes

def preprocess(training_path, window, fc_train):
    """
    This function carries out the signal processing, spike detection,
    and spike extraction procedures by making calls to the filter,
    detect, and extract functions.

    INPUTS
    * training_path:        path of training.mat
    * window:               list with the number of points to the left and right
                            of each peak that should be extracted
    * fc_train:             cut-off frequency for denoising training data

    OUTPUTS
    * train_spikes:         spike data points extracted from training recording
    * train_classes:        neuron type corresponding to each extracted spike
    """
    t = spio.loadmat(training_path, squeeze_me=True)

    Index = t['Index'] # The location in the recording (in samples) of each spike.
    Class = t['Class'] # The class (1, 2, 3 or 4), i.e the type of neuron that generated each spike.

    d_train = t['d'] # Raw time domain recording for training dataset

    # Denoise the training recording with a low-pass filter
    train_filtered = filter(d_train, fc_train, 'low')

    # Detect the positions of peaks within the training recording
    train_peaks = detect(train_filtered)

    # Extract the spikes from the training recording and getting the corresponding neuron class values
    train_spikes, train_classes = extract(train_filtered, train_peaks, window, Index, Class)

    return train_spikes, train_classes

def convert_to_one_hot(labels):
    """
    Converts vector of labels to to array of one-hot encoded vectors.
    
    INPUTS 
    * labels:   vector of labels

    OUTPUTS
    * encoded:  one-hot encoded array
    """
    # Convert label values from 1-4 to 0-3 for one-hot encoding
    labels = np.subtract(labels, 1)

    # Total number of classes in vector
    C = np.max(labels) + 1 

    # One-hot encoding of labels vector
    encoded = np.eye(C)[labels.reshape(-1)].T
    encoded = encoded.T

    return encoded

def build(n_x, n_y, n_h, l_r):
    """
    Building the structure of the neural network and compiling it.

    INPUTS
    * n_x:      number of input nodes
    * n_y:      number of output nodes
    * n_h:      number of hidden nodes
    * l_r:      learning rate

    OUTPUTS
    * model:    the compiled neural network model  
    """
    # Creating a sequential Keras model
    model = Sequential()

    # Adding input layer with n_x nodes
    model.add(Input(shape=n_x))
    # Adding single hidden layer with n_h hidden nodes
    model.add(Dense(n_h, activation='relu'))
    # Adding output layer with n_y nodes
    model.add(Dense(n_y, activation='softmax'))

    # Compiling the neural network with categorical crossentropy loss
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=l_r), metrics=['accuracy'])

    return model

training_path = 'datasets/training.mat' # Path of training data 

window = [15, 26] # Window size for extracting spikes from the recordings
fc_train = 2500 # Cut-off frequency for denoising training recording

# Detect and extract spikes and spike classes to create training set (X) and labels (y)
X, y = preprocess(training_path, window, fc_train)

# Convert the lists into numpy arrays
X, y = np.asarray(X), np.asarray(y)

# Convert neuron classes to one hot encoded vector representation
y = convert_to_one_hot(y)

# Split training dataset with an 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize parameters for neural network
n_x = X_train.shape[1] # Input nodes
n_y = y_train.shape[1] # Output nodes
n_h = 20 # Hidden nodes
l_r = 0.01 # Learning rate (default Adam optimizer value)
epochs = 50 # Total number of epochs
batch_size = 16 # Number of training examples to ues per iteration

# Build the neural network model
model = build(n_x, n_y, n_h, l_r)

# Train the model on the extracted spikes with their corresponding classes
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# Predict classes for spikes in the test subset
y_predict = model.predict(X_test, batch_size=batch_size)

# Convert the softmax probabilities into one-hot encoded vectors
y_predict = (y_predict == y_predict.max(axis=1)[:,None]).astype(int)

# Displaying performance metrics
performance_metrics = metrics.classification_report(y_test, y_predict, digits=4)
print(performance_metrics)
accuracy = metrics.accuracy_score(y_test, y_predict)
print("Accuracy = {}".format(np.round(accuracy,4)))