import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter

train_path = '/Users/saif/Desktop/University/Year 4/CI/Final/datasets/training.mat'
submission_path = '/Users/saif/Desktop/University/Year 4/CI/Final/datasets/submission.mat'

mat = spio.loadmat(train_path, squeeze_me=True)
d = mat['d'] # Raw time domain recording (1440000 samples), 25 kHz sampling frequency.
Index = mat['Index'] # The location in the recording (in samples) of each spike.
Class = mat['Class'] # The class (1, 2, 3 or 4), i.e the type of neuron that generated each spike.

mat = spio.loadmat(submission_path, squeeze_me=True)
# d = mat['d'] # Raw time domain recording (1440000 samples), 25 kHz sampling frequency.

fs = 25000 # sampling frequency
time = np.arange(0, np.size(d)) * 1/fs

def filter_data(data, low, high, fs, order=2):
    # Determine Nyquist frequency
    nyq = fs/2
    # Set bands
    low = low/nyq
    high = high/nyq
    # Calculate coefficients
    b, a = butter(order, [low, high], btype='band')
    # Filter signal
    filtered_data = lfilter(b, a, data)

    return filtered_data

d = filter_data(d, 50, 5000, fs)
noise_mad = np.median(np.absolute(d)/0.6745) 
thresh = 4 * noise_mad

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(time, d)
ax.plot([time[0], time[-1]], [thresh, thresh], 'r')
ax.set_title('Sampling Frequency: {}Hz'.format(fs))
ax.set_xlim(0, time[fs])
ax.set_xlabel('time [s]')
ax.set_ylabel('amplitude [mV]')
plt.show()

def detect(data, thresh, fs, time):

    tdx = time*fs

    return 1

def extract(data, index, fs, pre, post):

    waves = []
    pre_idx = int(pre*fs)
    post_idx = int(post*fs)

    for idx in index:
        if idx-pre_idx >= 0 and idx+post_idx <= len(data):
            wave = data[(idx-pre_idx):(idx+post_idx)]
            waves.append(wave)

    return np.stack(waves)

waves = extract(d, Index, fs, 0.001, 0.002)
print(waves.shape)



def feature():
    return 1

def cluster():
    return 1