import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, lfilter
from scipy.signal import find_peaks

train_path = '/Users/saif/Desktop/University/Year 4/CI/Final/datasets/training.mat'
submission_path = '/Users/saif/Desktop/University/Year 4/CI/Final/datasets/submission.mat'

mat = spio.loadmat(train_path, squeeze_me=True)
d = mat['d'] # Raw time domain recording (1440000 samples), 25 kHz sampling frequency.
Index = mat['Index'] # The location in the recording (in samples) of each spike.
Class = mat['Class'] # The class (1, 2, 3 or 4), i.e the type of neuron that generated each spike.

# mat = spio.loadmat(submission_path, squeeze_me=True)
# d = mat['d'] # Raw time domain recording (1440000 samples), 25 kHz sampling frequency.

fs = 25000 # Sampling frequency
time = np.arange(0, np.size(d)) * 1/fs # Recording time in seconds

def filter_data(signal, low, high, fs, order=2):
    
    nyq = 0.5 * fs
    
    low = low / nyq
    high = high / nyq

    b, a = butter(order, [low, high], 'bandpass', analog=False)
    y = filtfilt(b, a, signal)

    return y

# Filtered signal
d_f = filter_data(d, 50, 2750, fs)

# Calculate MAD and threshold for peak detection
noise_mad = np.median(np.absolute(d_f)/0.6745) 
thresh = 5 * noise_mad

# Plotting the unfiltered and filtered signals for the first second
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(time, d)
ax.plot(time, d_f)
ax.plot([time[0], time[-1]], [thresh, thresh], 'r')
ax.set_title('Sampling Frequency: {}Hz'.format(fs))
ax.set_xlim(0, time[fs])
ax.set_xlabel('time [s]')
ax.set_ylabel('amplitude [mV]')
plt.show()

def detect(signal, fs):

    spikes = find_peaks(signal, height=thresh)[0]
    timestamps = spikes / fs
    ranges = (0, time[-1])
    sr = timestamps[(timestamps >= ranges[0]) & (timestamps <= ranges[1])]

    return spikes, sr

spikes, sr = detect(d_f, fs)
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(time, d)
ax.plot(time, d_f)
ax.plot(sr, [thresh]*sr.shape[0], 'ro', ms=4)
ax.set_xlim(0, time[fs])
plt.show()

def extract(signal, spikes, fs, pre, post):

    waves = []
    pre_idx = int(pre*fs)
    post_idx = int(post*fs)

    for idx in spikes:
        if idx-pre_idx >= 0 and idx+post_idx <= len(signal):
            wave = signal[(idx-pre_idx):(idx+post_idx)]
            waves.append(wave)

    return np.stack(waves)

pre, post = 0.001, 0.002
waves = extract(d_f, spikes, fs, pre, post)
print(waves.shape)
 
def feature():
    return 1

def cluster():
    return 1