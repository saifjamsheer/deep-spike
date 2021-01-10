import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

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
    * signal:       raw time domain recordings (numpy array)

    OUTPUTS
    * peaks:        vector representing positions of peaks within the window
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
        # Preventing duplicates from being added
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

def main():

    training_path = 'training.mat' # Path of training data 

    window = [15, 26] # Window size for extracting spikes from the recordings
    fc_train = 2500 # Cut-off frequency for denoising training recording

    # Detect and extract spikes and spike classes to create training set (X) and labels (y)
    X, y = preprocess(training_path, window, fc_train)

    # Split training dataset with an 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    k = 5 # Number of neighbors for k-nn
    p = 2 # Euclidean distance metric

    # Build the k-nn classifier and fit it to the training subset
    model = KNeighborsClassifier(n_neighbors=k, p=p)
    model.fit(X_train, y_train)

    # Predict classes for spikes in the test subset
    y_predict = model.predict(X_test)

    # Display confusion matrix
    c_matrix = metrics.confusion_matrix(y_test, y_predict)
    print(c_matrix)

    # Display performance metrics
    performance_metrics = metrics.classification_report(y_test, y_predict, digits=4)
    print(performance_metrics)

if __name__ == "__main__":
    main()