import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile
from scipy.signal import find_peaks

# Load the audio file
audio_file = 'GI_GMF_B3_353_20140520_n.wav'
y, sr = librosa.load(audio_file)

# Compute the Fourier Transform
fft = np.fft.fft(y)
magnitude = np.abs(fft)
frequency = np.fft.fftfreq(len(magnitude), 1/sr)  # Updated frequency calculation

# Keep only positive frequencies (since the signal is real)
positive_frequencies = frequency[:len(frequency)//2]
magnitude = magnitude[:len(magnitude)//2]

# Plot the original audio signal
plt.figure(figsize=(10, 4))
plt.plot(np.arange(len(y)) / sr, y)
plt.title('Original Audio Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# Set Nyquist frequency limit (for visualization purposes only)
nyquist_limit = sr / 2

# Plot the Fourier Transform
plt.figure(figsize=(10, 4))
plt.plot(positive_frequencies, magnitude)
plt.title('Fourier Transform')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, nyquist_limit)  # Set the x-axis limit to the Nyquist frequency limit

# Set custom tick positions and labels for better readability
custom_ticks = np.arange(0, nyquist_limit, 500)  # Smaller tick interval
plt.xticks(custom_ticks, [f'{int(tick)}' for tick in custom_ticks])

plt.tight_layout()
plt.show()

# Inverse Fourier Transform
reconstructed_signal = np.fft.ifft(fft).real

# Plot the Reconstructed Audio Signal
plt.figure(figsize=(10, 4))
plt.plot(np.arange(len(reconstructed_signal)) / sr, reconstructed_signal)
plt.title('Reconstructed Audio Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# Trying methods for analysis and evaluation of the fourier transform of the audio signal.
'''
# Harmonic Analysis
harmonic_indices = np.where((positive_frequencies > 0) & (positive_frequencies < nyquist_limit / 2))
noise_indices = np.where((positive_frequencies >= nyquist_limit / 2) & (positive_frequencies < nyquist_limit))

hnr = np.sum(magnitude[harmonic_indices]) / np.sum(magnitude[noise_indices])
print(f'Harmonic-to-Noise Ratio (HNR): {hnr}')

# Spectral Shape Measures
centroid = np.sum(positive_frequencies * magnitude) / np.sum(magnitude)
bandwidth = np.sum(magnitude * (positive_frequencies - centroid)**2) / np.sum(magnitude)

print(f'Spectral Centroid: {centroid} Hz')
print(f'Spectral Bandwidth: {bandwidth} Hz')

# Spectral Flatness
flatness = np.exp(np.mean(np.log(magnitude))) / (np.mean(magnitude))
print(f'Spectral Flatness: {flatness}')

# Peak Detection
peak_indices, _ = find_peaks(magnitude, height=0)
peak_frequencies = positive_frequencies[peak_indices]
peak_magnitudes = magnitude[peak_indices]

# Signal-to-Noise Ratio (SNR)
signal_power = np.sum(magnitude[harmonic_indices])
noise_power = np.sum(magnitude[noise_indices])
snr = 10 * np.log10(signal_power / noise_power)
print(f'Signal-to-Noise Ratio (SNR): {snr} dB')

# Frequency Analysis Bins
frequency_bins = np.linspace(0, nyquist_limit, 10)  # Adjust the number of bins as needed
bin_indices = np.digitize(positive_frequencies, frequency_bins)
bin_energies = [np.sum(magnitude[bin_indices == i]) for i in range(1, len(frequency_bins))]

# Plot the Frequency Analysis Bins
plt.figure(figsize=(10, 4))
plt.bar(frequency_bins[:-1], bin_energies, width=np.diff(frequency_bins), align='edge')
plt.title('Frequency Analysis Bins')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Energy')
plt.tight_layout()
plt.show()
'''

# Evaluation: The Fourier Transform of the input audio signal has observable "folding" or "aliasing" in the frequency domain.
# This is caused by the presence of frequency components that exist beyond the Nyquist frequency.
# The Nyquist frequency is the highest frequency that can be accurately represented by a digital signal, the value of which is half of the sampling rate of the signal.
# Frequencies that exist beyond the Nyquist frequency may cause aliasing. Aliasing occurs when frequencies above the Nyquist frequency are folded back into the audible frequency range.
# The solution to this aliasing issue would be to bandlimit the input signal or apply anti-aliasing filters, however, as can be seen in the time-domain representation of the reconstructed audio, the aliasing does not seem to have a visually observable effect.
# Thus, a Nyquist frequency limit of 22.05 kHz (half of the 44.1 kHz sampling rate) is sufficient for simply visualizing the Fourier Transform of the input audio signal.
# The reconstructed audio signal appears and sounds to be identical to the original audio signal, indicating that the inverse Fourier Transform was successful in reconstructing the original signal from its Fourier Transform.
# The Fourier Transform shows three distinct dominant frequency components.
# The first dominant frequency component is around 0-250Hz, the second dominant frequency component is around 500HZ, and the third dominant frequency component is around 1kHz.
# Note that the dominant frequency component at 0-250Hz may represent a fundamental frequency with harmonics or overtones.
# Similar traits can be observed in the dominant frequency component at 500Hz, although not as much, which may also represent a fundamental frequency with harmonics or overtones.
# Additionally, there are other frequency components present in the signal, but they are not as dominant as the three mentioned, notably at 1kHz and 2kHz.


