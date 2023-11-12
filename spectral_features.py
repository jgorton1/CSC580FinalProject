import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

#sample_rate, samples = wavfile.read('Data\\train\\jazz\\jazz.00005.wav')
def spectral_centroid(samples, sample_rate, npersegment):
    # computes spectral centroid for each time interval of stft
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=npersegment)
    # centroid is weighted sum of intensity and frequencies
    # divided by sum of intensities
    # for each time interval
    print(spectrogram.shape)
    print(frequencies.shape)
    centroids = np.divide(np.matmul(spectrogram.T,frequencies),np.sum(spectrogram,axis=0))
    return centroids

#sample_rate, samples = wavfile.read('Data\\train\\jazz\\jazz.00005.wav')
#print(spectral_centroid(samples[0:22050], sample_rate,int(0.05 * 22050)))

