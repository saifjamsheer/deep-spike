import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.decomposition import PCA
import random

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

def detect(signal, prominence=0):
    """
    Detects the peaks within the inputted signal and returns the positions
    of these peaks.

    INPUTS
    * signal:       raw time domain recordings (numpy array)
    * prominence:   float that determines prominence of detected peaks

    OUTPUTS
    * peaks:        vector representing positions of peaks within the window
    """
    # Calculate median absolute deviation of the points in the signal
    mad = np.median(np.absolute(signal)/0.6745) 
    # Set a threshold for peak detection
    threshold = 5 * mad
    # Use find_peaks to find positions of peaks in the signal
    peaks, _ = find_peaks(signal, height=threshold, prominence=prominence)

    return peaks

def extract(signal, peaks, window, type, idx=0, neurons=0):
    """
    Extracts data points distributed around spike peaks based on the
    window size.

    INPUTS
    * signal:   raw time domain recordings (numpy array)
    * peaks:    vector representing positions of peaks within the signal
    * window:   list with the number of points to the left and right
                of each peak that should be extracted
    * type:     string representing whether the inputted signal is part 
                of the training recording or the submission recording
    * idx:      Index vector from training.mat
    * neurons:  Class vector from training.mat

    OUTPUTS
    * spikes:   Array of extracted sample points
    * classes:  Array of corresponding neuron types
    """
    spikes = [] # Initialize array of sample points
    
    if type == 'training':

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
    
    if type == 'submission':
        # Iterate through the positions of peaks
        for peak_idx in peaks:
            # Extract data points to the left and right of a peak
            values = signal[peak_idx-window[0]:peak_idx+window[1]]

            spikes.append(values)
            
        return spikes
    
    # Returns an empty list in case of a type mismatch
    return spikes

def preprocess(training_path, submission_path, window, fc_train, fc_normalize, fc_submission):
    """
    This function carries out the signal processing, spike detection,
    and spike extraction procedures by making calls to the filter,
    detect, and extract functions.

    INPUTS
    * training_path:        path of training.mat
    * submission_path:      path of submission.mat
    * window:               list with the number of points to the left and right
                            of each peak that should be extracted
    * fc_train:             cut-off frequency for denoising training data
    * fc_normalize:         cut-off frequency for normalizing submission data
    * fc_submission:        cut-off frequency for denoising submission data

    OUTPUTS
    * train_spikes:         spike data points extracted from training recording
    * train_classes:        neuron type corresponding to each extracted spike
    * submission_spikes:    spike data points extracted from submission recording
    * submission_peaks:     location of spikes within the submission recording
    """
    t = spio.loadmat(training_path, squeeze_me=True)
    s = spio.loadmat(submission_path, squeeze_me=True)

    Index = t['Index'] # The location in the recording (in samples) of each spike.
    Class = t['Class'] # The class (1, 2, 3 or 4), i.e the type of neuron that generated each spike.

    d_train = t['d'] # Raw time domain recording for training dataset
    d_submission = s['d'] # Raw time domain recording for submission dataset

    # Denoise the training recording with a low-pass filter
    train_filtered = filter(d_train, fc_train, 'low')

    # Detect the positions of peaks within the training recording
    train_peaks = detect(train_filtered)

    # Extract the spikes from the training recording and getting the corresponding neuron class values
    train_spikes, train_classes = extract(train_filtered, train_peaks, window, 'training', Index, Class)

    # Normalize submission recording singal
    submission_normalized = filter(d_submission, fc_normalize, 'high')

    # Denoise the training recording with a low-pass filter
    submission_filtered = filter(submission_normalized, fc_submission, 'low')
    
    # Detect the positions of peaks within the training recording
    submission_peaks = detect(submission_filtered, prominence=0.2)

    # Extract the spikes from the submission recording
    submission_spikes = extract(submission_filtered, submission_peaks, window, 'submission')

    return train_spikes, train_classes, submission_peaks, submission_spikes

def optimize(solution, demand, alpha, iterations, var, inputs, labels):
    """
    Find the optimal parameters for the k-nearest neighbors algorithm
    through simulated annealing.

    INPUTS:
    * solution:     initial parameters
    * demand:       target performance value
    * alpha:        temperature reduction rate
    * iterations:   total iterations for the optimization algorithm
    * var:          constant for altering solutions
    * inputs:       inputs for the k-nn algorithm
    * labels:       labels corresponding to the inputs

    OUTPUTS:
    * k_optimal:    optimal number of nearest neighbors (integer greater than 0)
    * p_optimal:    optimal distance metric (integer between 1 and 3, inclusive)
    """

    # Reduce dimensions with PCA to speed up computation
    pca = PCA(n_components=0.99) 
    inputs = pca.fit_transform(inputs)
    
    # Simulated annealing to find the optimal paramaters
    solution = anneal(solution, demand, alpha, iterations, var, inputs, labels)
    k_optimal = int(round(solution[0]))
    p_optimal = int(round(solution[1]))

    return k_optimal, p_optimal

def k_n_n(X, y, k, p):
    """
    Fits the k-nn model to a training subset and returns
    the cross-validation sets.

    INPUTS
    * X:        training set
    * y:        training labels
    * k:        number of nearest neighbors
    * p:        distance metric

    OUTPUTS
    * model:    k-nn classifier fit to the training data
    * X_val:    validation set inputs
    * y_val:    validation set labels
    """

    # Split input into training subset and validation subset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)

    # Build the k-nn classifier and fit it to the training subset
    model = KNeighborsClassifier(n_neighbors=k, p=p)
    model.fit(X_train, y_train)

    return model, X_val, y_val

def score(model, X_val, y_val):
    """
    Calculates the accuracy of the model on the validation
    dataset for paramater tuning.

    INPUTS
    * model:    k-nn classifier
    * X_val:    validation set inputs
    * y_val:    validation set labels

    OUTPUT
    * accuracy: accuracy of the classifier
    """
    # Predict classes for spikes in the validation subset
    predictions = model.predict(X_val)

    accuracy = metrics.accuracy_score(y_val, predictions)

    return accuracy

def acceptance_probability(old_cost, new_cost, T):
    """
    Calculates acceptance proability of new solutions to determine
    if the new solution should be accepted.

    *** TAKEN FROM LAB6 ***
    """
    a = np.exp((old_cost-new_cost)/T)
    return a

def anneal(solution, demand, alpha, iterations, var, inputs, labels):
    """
    Implementation of the simulated annealing algorithm for optimization.

    *** TAKEN FROM LAB6 ***
    """
    old_cost = cost(solution, demand, inputs, labels)
    cost_values = list()
    cost_values.append(old_cost)
    T = 1.0
    T_min = 0.001
    while T > T_min:
        i = 1
        while i <= iterations:
            new_solution = neighbor(solution, var)
            new_cost = cost(new_solution, demand, inputs, labels)
            ap = acceptance_probability(old_cost, new_cost, T)
            if ap > random.random():
                solution = new_solution
                old_cost = new_cost
            i += 1
            cost_values.append(old_cost)
        T = T * alpha
    return solution

def cost(supply, demand, inputs, labels):
    """
    Cost function for simulated annealing. Calculates the difference
    between the target accuracy and the accuracy of the classifier.

    INPUTS
    * supply:   current parameters ([k, p])
    * demand:   target accuracy
    * inputs:   training inputs (i.e. X_train)
    * labels:   training labels (i.e. y_train)

    OUTPUTS
    * cost:     cost of the current solution
    """
    k = int(round(supply[0]))
    p = int(round(supply[1])) 
    # Build a k-nn classifier
    model, X_val, y_val = k_n_n(inputs, labels, k, p)
    # Calculate the accuracy of the classifier on the validation set
    performance = score(model, X_val, y_val)
    dcost = demand - performance
    return dcost

def neighbor(solution, d):
    """
    Calculates new parameter values neighboring the previous ones.

    INPUTS:
    * solution: current parameters ([k, p])
    * d:        constant for creating new solutions

    OUTPUTS:
    * solution: new parameters ([k, p])
    """

    delta = np.random.random((2,1))
    scale = np.full((2,1), 2*d)
    offset = np.full((2,1), 1.0-d)
    
    var = np.multiply(delta, scale)
    m = np.add(var, offset)

    # Generate new solutions
    solution[0] = solution[0] * m[0][0]
    solution[1] = np.random.randint(1,3)

    if solution[0] < 0.5:
        solution[0] = 1

    return solution

def main():

    training_path = 'training.mat' # Path of training data 
    submission_path = 'submission.mat' # Path of submission data

    window = [15, 26] # Window size for extracting spikes from the recordings
    fc_train = 2500 # Cut-off frequency for denoising training recording
    fc_normalize = 25 # Cut-off frequency for normalizing submission recording
    fc_submission = 1900 # Cut-off frequency for denoising submission recording

    # Detect and extract training spikes, training spike classes, submission spike
    # positions (index values), and submission spikes
    train_spikes, train_classes, submission_peaks, submission_spikes = preprocess(training_path, submission_path, window, fc_train, fc_normalize, fc_submission)

    # Assign the returned output values to new variables
    X, y, S = train_spikes, train_classes, submission_spikes
    Index_S = submission_peaks # Index vector for submission

    # Split training dataset with an 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Initialize simulated annealing variables
    alpha = 0.1 # Temperature decay rate
    iterations = 100 # Number of iterations for simulated annealing
    var = 0.25 # Constant for calculating new solutions
    solution = [np.random.randint(1,20), np.random.randint(1,3)] # Initial parameters ([k, p])
    demand = 1.0 # Target performance value

    # Determine the optimal parameters for the k-nearest neighbors algorithm
    k_optimal, p_optimal = optimize(solution, demand, alpha, iterations, var, X_train, y_train)
    print("Optimal parameters: n_neighbors = {}, p = {}".format(k_optimal, p_optimal))

    # Build the k-nn classifier and fit it to the training subset
    model = KNeighborsClassifier(n_neighbors=k_optimal, p=p_optimal)
    model.fit(X_train, y_train)

    # Predict classes for spikes in the test subset
    y_predict = model.predict(X_test)

    # Display confusion matrix
    c_matrix = metrics.confusion_matrix(y_test, y_predict)
    print(c_matrix)

    # Display performance metrics
    performance_metrics = metrics.classification_report(y_test, y_predict, digits=4)
    print(performance_metrics)

    # Predict classes for spikes extracted from submission recording
    Class_S = model.predict(S)

    # Export submission data as .mat file
    spio.savemat('13235.mat', mdict={'Index': Index_S, 'Class': Class_S})

if __name__ == "__main__":
    main()