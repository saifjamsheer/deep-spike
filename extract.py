import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, lfilter
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plot as pl
import cluster as cl

def filter_data(signal, low, high, fs, order=2):
    
    nyq = 0.5 * fs
    
    low = low / nyq
    high = high / nyq

    b, a = butter(order, [low, high], 'bandpass', analog=False)
    y = filtfilt(b, a, signal)
    return y

def detect(signal, index, thresh, fs, type):

    if type == 'train':
        spikes = index
    else:
        spikes = find_peaks(signal, height=thresh)[0]

    return spikes

def extract(signal, spikes, fs, pre, post):

    waves = []
    pre_idx = int(pre*fs)
    post_idx = int(post*fs)

    for idx in spikes:
        if idx-pre_idx >= 0 and idx+post_idx <= len(signal):
            wave = signal[(idx-pre_idx):(idx+post_idx)]
            waves.append(wave)

    return np.stack(waves)

def main():

    train_path = '/Users/saif/Desktop/University/Year 4/CI/Final/datasets/training.mat'
    submission_path = '/Users/saif/Desktop/University/Year 4/CI/Final/datasets/submission.mat'

    type = 'train'

    tmat = spio.loadmat(train_path, squeeze_me=True)
    smat = spio.loadmat(submission_path, squeeze_me=True)

    Index = tmat['Index'] # The location in the recording (in samples) of each spike.
    Class = tmat['Class'] # The class (1, 2, 3 or 4), i.e the type of neuron that generated each spike.

    if type == 'train':
        d = tmat['d'] # Raw time domain recording (1440000 samples), 25 kHz sampling frequency.
    else:
        d = smat['d'] # Raw time domain recording (1440000 samples), 25 kHz sampling frequency.

    fs = 25000 # Sampling frequency
    time = np.arange(0, np.size(d)) * 1/fs # Recording time in seconds

    # Filtered signal
    d_f = filter_data(d, 50, 2750, fs)

    # Calculate MAD and threshold for peak detection
    noise_mad = np.median(np.absolute(d_f)/0.6745) 
    thresh = 5 * noise_mad

    # Plotting the unfiltered and filtered signals for the first second
    pl.signals(d, d_f, time, thresh, fs)

    spikes = detect(d_f, Index, thresh, fs, type)

    # Plotting the unfiltered and filtered signals for the first second with the peaks
    pl.peaks(d, d_f, time, thresh, spikes, fs)

    pre, post = 0.001, 0.002
    waves = extract(d_f, spikes, fs, pre, post)
    print(waves.shape)

    data = np.asarray(waves)
    np.save('inputs.npy', data)
    
    pl.waveforms(waves, pre, post, fs, n=100)

    # Dimensionality Reduction & K-Means Clustering
    # n_c = cl.pca_components(waves)
    # scaler = StandardScaler()
    # reduced = cl.reduce(waves, scaler, n_c)
    # n_clusters = np.unique(Class).shape[0]
    # clusters, centers = cl.clustering(reduced, n_clusters, 300)
    # pl.clusters(reduced, clusters, n_clusters, centers, waves, fs, time)

if __name__ == "__main__":
    main()

