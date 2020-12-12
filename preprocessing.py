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

# print(sorted(Index, reverse=True)[0:10])

# mat = spio.loadmat(submission_path, squeeze_me=True)
# d = mat['d'] # Raw time domain recording (1440000 samples), 25 kHz sampling frequency.

fs = 25000 # Sampling frequency
time = np.arange(0, np.size(d)) * 1/fs

def filter_data(signal, low, high, fs, order=2):
    
    nyq = 0.5 * fs
    
    low = low / nyq
    high = high / nyq

    b, a = butter(order, [low, high], 'bandpass', analog=False)
    y = filtfilt(b, a, signal)

    return y

d2 = filter_data(d, 10, 1200, fs)
noise_mad = np.median(np.absolute(d2)/0.6745) 
thresh = 5 * noise_mad

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(time, d)
ax.plot(time, d2)
ax.plot([time[0], time[-1]], [thresh, thresh], 'r')
ax.set_title('Sampling Frequency: {}Hz'.format(fs))
# ax.set_xlim(0, time[fs])
ax.set_xlim(57.570, 57.595)
ax.set_ylim(-2.5, 4.5)
ax.set_xlabel('time [s]')
ax.set_ylabel('amplitude [mV]')
plt.show()

spikes = find_peaks(d2, height=thresh)[0]
print(len(spikes))
timestamps = spikes / fs
range_in_s = (0, time[-1])
spikes_in_range = timestamps[(timestamps >= range_in_s[0]) & (timestamps <= range_in_s[1])]
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(time, d)
ax.plot(time, d2)
ax.plot(spikes_in_range, [thresh]*spikes_in_range.shape[0], 'ro', ms=2)
# ax.set_xlim(0, time[fs])
# ax.set_xlim(57.570, 57.595)
# ax.set_ylim(-2.5, 4.5)
plt.show()

# plt.scatter([x for x, y in peaks], [y for x, y in peaks], color = 'red')
# plt.plot(time, d, 'k')
# plt.show()

# def detect(data, thresh, fs, time):

#     tdx = time*fs

#     crossings = np.diff((data >= thresh).astype(int) != 0).nonzero()[0]
#     sufficient = np.insert(np.diff(crossings) >= tdx, 0, True)
    
#     while not np.all(sufficient):
#         crossings = crossings[sufficient]
#         print(sufficient)
#         sufficient = np.insert(np.diff(crossings) >= tdx, 0, True)

#     return crossings
 
# crossings = detect(d2, thresh, fs, 0.0012)

# print(crossings)
# print(len(crossings))
# print(np.array(sorted(Index)))
# print(len(Index))

# def get_next_minimum(signal, index, max_samples_to_search):
#     """
#     Returns the index of the next minimum in the signal after an index

#     :param signal: The signal as a 1-dimensional numpy array
#     :param index: The scalar index
#     :param max_samples_to_search: The number of samples to search for a minimum after the index
#     """
#     search_end_idx = min(index + max_samples_to_search, signal.shape[0])
#     min_idx = np.argmax(signal[index:search_end_idx])
#     return index + min_idx

# def align_to_minimum(signal, fs, threshold_crossings, search_range):
#     """
#     Returns the index of the next negative spike peak for all threshold crossings

#     :param signal: The signal as a 1-dimensional numpy array
#     :param fs: The sampling frequency in Hz
#     :param threshold_crossings: The array of indices where the signal crossed the detection threshold
#     :param search_range: The maximum duration in seconds to search for the minimum after each crossing
#     """
#     search_end = int(search_range*fs)
#     aligned_spikes = [get_next_minimum(signal, t, search_end) for t in threshold_crossings]
#     return np.array(aligned_spikes)

# spikes = align_to_minimum(d2, fs, crossings, 0.0005)
# # spikes = crossings
# print(spikes)

# timestamps = spikes / fs
# range_in_s = (0, time[-1])
# spikes_in_range = timestamps[(timestamps >= range_in_s[0]) & (timestamps <= range_in_s[1])]

# fig, ax = plt.subplots(figsize=(15, 5))
# ax.plot(time, d)
# ax.plot(time, d2)
# ax.plot(spikes_in_range, [thresh]*spikes_in_range.shape[0], 'ro', ms=2)
# ax.set_xlim(0, time[fs])
# # ax.set_xlim(57.570, 57.595)
# # ax.set_ylim(-2.5, 4.5)
# plt.show()

# def extract(data, index, fs, pre, post):

#     waves = []
#     pre_idx = int(pre*fs)
#     post_idx = int(post*fs)

#     for idx in index:
#         if idx-pre_idx >= 0 and idx+post_idx <= len(data):
#             wave = data[(idx-pre_idx):(idx+post_idx)]
#             waves.append(wave)

#     return np.stack(waves)

# waves = extract(d, Index, fs, 0.001, 0.002)
# print(waves.shape)



def feature():
    return 1

def cluster():
    return 1