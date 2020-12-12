import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np

train_path = '/Users/saif/Desktop/University/Year 4/CI/Final/datasets/training.mat'
submission_path = '/Users/saif/Desktop/University/Year 4/CI/Final/datasets/submission.mat'

mat = spio.loadmat(train_path, squeeze_me=True)
d = mat['d'] # Raw time domain recording (1440000 samples), 25 kHz sampling frequency.
Index = mat['Index'] # The location in the recording (in samples) of each spike.
Class = mat['Class'] # The class (1, 2, 3 or 4), i.e the type of neuron that generated each spike.

sf = 25000 # sampling frequency
sec = len(d)/sf # duration of recording in seconds
time = np.linspace(0, sec, len(d))

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(time[0:sf], d[0:sf])
ax.set_title('Sampling Frequency: {}Hz'.format(sf))
ax.set_xlim(0, time[sf])
ax.set_xlabel('time [s]')
ax.set_ylabel('amplitude [mV]')
plt.show()

def extract(data, window, tf, offset, ):
    return 1

def feature():
    return 1

def cluster():
    return 1