import matplotlib.pyplot as plt
import numpy as np

def signals(unfiltered, filtered, time, thresh, fs):

    _, ax = plt.subplots(figsize=(15, 5))
    ax.plot(time, unfiltered, c='lightcoral')
    ax.plot(time, filtered, c='crimson')
    ax.plot([time[0], time[-1]], [thresh, thresh], linewidth=2.5, color='darkmagenta')
    ax.set_title('Sampling Frequency: {}Hz'.format(fs))
    ax.set_xlim(0, time[fs])
    ax.set_xlabel('time [s]')
    ax.set_ylabel('amplitude [mV]')
    plt.show()

def peaks(unfiltered, filtered, time, thresh, spikes, fs):

    _, ax = plt.subplots(figsize=(15, 5))
    ax.plot(time, unfiltered, c='lightcoral')
    ax.plot(time, filtered, c='crimson')

    timestamps = spikes / fs
    ranges = (0, time[-1])
    sr = timestamps[(timestamps >= ranges[0]) & (timestamps <= ranges[1])]

    ax.plot(sr, [thresh]*sr.shape[0], c='darkmagenta', marker='o', ms=4, linestyle='None')
    ax.set_xlim(0, time[fs])
    plt.show()

def waveforms(waves, pre, post, fs, n=100):

    time = np.arange(-pre*1000, post*1000, 1000/fs)
    _, ax = plt.subplots(figsize=(12, 6))

    for _ in range(n):
        spike = np.random.randint(0, waves.shape[0])
        ax.plot(time, waves[spike, :], color='k', linewidth=1, alpha=0.3)

    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('amplitude [mV]')
    ax.set_title('Spike Waveforms')
    plt.show()