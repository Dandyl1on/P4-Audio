import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks

# Load audio file
file_path = r'C:\Users\Tze Huo Gucci Ho\Desktop\Git Projects\P4-Audio\GI_GMF_B3_353_20140520_n.wav'
sample_rate, data = wavfile.read(file_path)

# Compute the Fourier Transform
frequencies = np.fft.fftfreq(len(data), d=1/sample_rate)
spectrum = np.fft.fft(data)

# Plot the magnitude spectrum (zoomed to full range)
plt.figure(figsize=(12, 6))
plt.plot(frequencies, np.abs(spectrum))
plt.title('Fourier Transform of Audio Signal (Full Range)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 2100)  # Set the x-axis limits to cover the full range
plt.grid(True)
plt.show()

# Find peaks in the spectrum
peaks, _ = find_peaks(np.abs(spectrum))

# Extract dominant frequencies
dominant_frequencies = frequencies[peaks]
print("Dominant Frequencies (Hz):", dominant_frequencies)

# Calculate spectral centroid
spectral_centroid = np.sum(np.abs(spectrum) * frequencies) / np.sum(np.abs(spectrum))
print("Spectral Centroid (Hz):", spectral_centroid)

# Extract phase information
phases = np.angle(spectrum)
print("Phase Information:", phases)
