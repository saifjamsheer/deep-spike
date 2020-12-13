import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, lfilter
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

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
ax.plot(time, d, c='lightcoral')
ax.plot(time, d_f, c='crimson')
ax.plot([time[0], time[-1]], [thresh, thresh], linewidth=2.5, color='darkmagenta')
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
ax.plot(time, d, c='lightcoral')
ax.plot(time, d_f, c='crimson')
ax.plot(sr, [thresh]*sr.shape[0], c='darkmagenta', marker='o', ms=4, linestyle='None')
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

pre, post = 0.001, 0.001
waves = extract(d_f, spikes, fs, pre, post)
print(waves.shape)

def waveforms(waves, pre, post, n=100):

    time = np.arange(-pre*1000, post*1000, 1000/fs)
    _, ax = plt.subplots(figsize=(12, 6))

    for i in range(n):
        spike = np.random.randint(0, waves.shape[0])
        ax.plot(time, waves[spike, :], color='k', linewidth=1, alpha=0.3)

    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('amplitude [mV]')
    ax.set_title('Spike Waveforms')
    plt.show()

waveforms(waves, pre, post)

def pca_components(waves):

    pca = PCA()
    waves = pca.fit_transform(waves)
    t = sum(pca.explained_variance_)
    n_c = 0 
    variance = 0
    while variance/t < 0.90:
        variance += pca.explained_variance_[n_c]
        n_c += 1
    
    return n_c

n_c = pca_components(waves)

def reduce(waves, scaler, n_c):

    scaled = scaler.fit_transform(waves)
    pca = PCA(n_components=n_c)
    reduced = pca.fit_transform(scaled)

    return reduced

scaler = StandardScaler() # or MinMaxScaler
reduced = reduce(waves, scaler, n_c)

def pcaplot(reduced, type):

    if type == 1:
        _, axs = plt.subplots(2, 3, figsize=(10,6))
        axs[0,0].plot(reduced[:,0], reduced[:,1], '.', c='midnightblue')
        axs[0,0].set_xlabel('Principal Component 1')
        axs[0,0].set_ylabel('Principal Component 2')
        axs[0,0].set_title('PCA1 v PC2')
        axs[0,1].plot(reduced[:,0], reduced[:,2], '.', c='royalblue')
        axs[0,1].set_xlabel('Principal Component 1')
        axs[0,1].set_ylabel('Principal Component 3')
        axs[0,1].set_title('PCA1 v PC3')
        axs[0,2].plot(reduced[:,0], reduced[:,3], '.', c='dodgerblue')
        axs[0,2].set_xlabel('Principal Component 1')
        axs[0,2].set_ylabel('Principal Component 4')
        axs[0,2].set_title('PCA1 v PC4')
        axs[1,0].plot(reduced[:,1], reduced[:,2], '.', c='steelblue')
        axs[1,0].set_xlabel('Principal Component 2')
        axs[1,0].set_ylabel('Principal Component 3')
        axs[1,0].set_title('PCA2 v PC3')
        axs[1,1].plot(reduced[:,1], reduced[:,3], '.', c='deepskyblue')
        axs[1,1].set_xlabel('Principal Component 2')
        axs[1,1].set_ylabel('Principal Component 4')
        axs[1,1].set_title('PCA2 v PC4')
        axs[1,2].plot(reduced[:,2], reduced[:,3], '.', c='mediumblue')
        axs[1,2].set_xlabel('Principal Component 3')
        axs[1,2].set_ylabel('Principal Component 4')
        axs[1,2].set_title('PCA3 v PC4')
        plt.tight_layout()
        plt.show()

    if type == 2:
        fig, axs = plt.subplots(1, 4, figsize=(15, 4), squeeze=False)
        axs[0,0].scatter(reduced[:, 0], reduced[:, 1], c=reduced[:, 2])
        axs[0,0].set_xlabel('PCA1')
        axs[0,0].set_ylabel('PCA2')
        axs[0,0].set_title('Color = PCA3')
        axs[0,1].scatter(reduced[:, 0], reduced[:, 1], c=reduced[:, 3])
        axs[0,1].set_xlabel('PCA1')
        axs[0,1].set_ylabel('PCA2')
        axs[0,1].set_title('Color = PCA4')
        axs[0,2].scatter(reduced[:, 0], reduced[:, 2], c=reduced[:, 3])
        axs[0,2].set_xlabel('PCA1')
        axs[0,2].set_ylabel('PCA3')
        axs[0,2].set_title('Color = PCA4')
        axs[0,3].scatter(reduced[:, 1], reduced[:, 2], c=reduced[:, 3])
        axs[0,3].set_xlabel('PCA2')
        axs[0,3].set_ylabel('PCA3')
        axs[0,3].set_title('Color = PCA4')
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()

pcaplot(reduced, 1)

def feature():
    return 1

def cluster():
    return 1