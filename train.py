import scipy.io as spio

train_path = '/Users/saif/Desktop/University/Year 4/CI/Final/datasets/training.mat'
submission_path = '/Users/saif/Desktop/University/Year 4/CI/Final/datasets/submission.mat'

mat = spio.loadmat(train_path, squeeze_me=True)
d = mat['d'] # Raw time domain recording (1440000 samples), 25 kHz sampling frequency.
Index = mat['Index'] # The location in the recording (in samples) of each spike.
Class = mat['Class'] # The class (1, 2, 3 or 4), i.e the type of neuron that generated each spike.

