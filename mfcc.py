import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

def mel_filter_bank(num_filters, fft_size, sample_rate):
    # Define the Mel scale range (in Hz)
    mel_low = 0
    mel_high = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_points = np.linspace(mel_low, mel_high, num_filters + 2)

    # Convert Mel scale points to Hz
    hz_points = 700 * (10**(mel_points / 2595) - 1)

    # Convert Hz points to FFT bins
    fft_bins = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)

    # Create the triangular filter bank
    filters = np.zeros((num_filters, fft_size // 2 + 1))
    for i in range(1, num_filters + 1):
        filters[i - 1, fft_bins[i - 1]:fft_bins[i]] = (
            np.arange(fft_bins[i - 1], fft_bins[i]) - fft_bins[i - 1]
        ) / (fft_bins[i] - fft_bins[i - 1])
        filters[i - 1, fft_bins[i]:fft_bins[i + 1]] = (
            fft_bins[i + 1] - np.arange(fft_bins[i], fft_bins[i + 1])
        ) / (fft_bins[i + 1] - fft_bins[i])

    return filters

# Load an example audio file
sample_rate, data = wavfile.read('Data\\train\\jazz\\jazz.00005.wav')

# Compute the spectrogram
frequencies, times, Sxx = spectrogram(data, fs=sample_rate, nperseg=256)

# Define parameters
num_filters = 26  # Number of Mel filters
fft_size = 256  # FFT size

# Create the Mel filter bank
mel_filters = mel_filter_bank(num_filters, fft_size, sample_rate)

# Apply the Mel filter bank to the spectrogram
mel_spectrum = np.dot(mel_filters, Sxx)

# Plot the original spectrogram and the Mel spectrogram
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='auto')
plt.title('Original Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar()

plt.subplot(2, 1, 2)
plt.pcolormesh(times, np.linspace(0, sample_rate / 2, num_filters), 10 * np.log10(mel_spectrum), shading='auto')
plt.title('Mel Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Mel Frequency')
plt.colorbar()

plt.tight_layout()
plt.show()