import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

sample_rate, samples = wavfile.read('Data\\train\\jazz\\jazz.00005.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=66150)

print(frequencies)

plt.pcolormesh(times, frequencies, np.log(spectrogram))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
