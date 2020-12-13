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

def clusters(data, clusters, n_clusters, centers, waves, fs, time):

    _, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].scatter(data[:, 0], data[:, 1], c=clusters)
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[0].set_title('Clustered Data')

    time = np.linspace(0, waves.shape[1]/fs, waves.shape[1])*1000
    for i in range(n_clusters):
        cluster_mean = waves[clusters==i, :].mean(axis=0)
        cluster_std = waves[clusters==i, :].std(axis=0)

        ax[1].plot(time, cluster_mean, label='Cluster {}'.format(i))
        ax[1].fill_between(time, cluster_mean-cluster_std, cluster_mean+cluster_std, alpha=0.15)

    ax[1].set_title('Average Waveforms')
    ax[1].set_xlim([0, time[-1]])
    ax[1].set_xlabel('time [ms]')
    ax[1].set_ylabel('amplitude [mV]')

    plt.legend()
    plt.show()