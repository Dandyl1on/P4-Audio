import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile
from scipy.signal import find_peaks

# Load the audio file
audio_file = 'GI_GMF_B3_353_20140520_n.wav'
y, sr = librosa.load(audio_file)

# TASK 1: Evaluate and represent the Fourier Transform of the input audio signal.

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

# Harmonic Analysis
harmonic_indices = np.where((positive_frequencies > 0) & (positive_frequencies < nyquist_limit / 2))
noise_indices = np.where((positive_frequencies >= nyquist_limit / 2) & (positive_frequencies < nyquist_limit))

hnr = np.sum(magnitude[harmonic_indices]) / np.sum(magnitude[noise_indices])
print(f'Harmonic-to-Noise Ratio (HNR): {hnr}')

# Find peaks in the magnitude spectrum with a height threshold
# Here, we set the threshold to be 5% of the maximum magnitude
your_threshold_value = 0.05 * np.max(magnitude)
peaks, _ = find_peaks(magnitude, height=your_threshold_value)

# Sort peaks by magnitude in ascending order
sorted_peak_indices = np.argsort(magnitude[peaks])

# Select the top 5 peaks
top_sorted_peak_indices = sorted_peak_indices[-5:]

# Print the top 5 dominant frequencies and their amplitudes (from lowest to highest)
top_frequencies = positive_frequencies[peaks[top_sorted_peak_indices]]
top_amplitudes = magnitude[peaks[top_sorted_peak_indices]]

# Combine frequencies and amplitudes into a list of tuples
dominant_info = list(zip(top_frequencies, top_amplitudes))

# Sort the list by amplitude in ascending order
dominant_info.sort(key=lambda x: x[1])

# Print the sorted dominant frequencies and their amplitudes
for freq, amp in dominant_info:
    print(f'Dominant Frequency: {freq} Hz, Amplitude: {amp}')

# Plot the identified peaks on the Fourier Transform plot
plt.figure(figsize=(10, 4))
plt.plot(positive_frequencies, magnitude)
plt.plot(positive_frequencies[peaks[top_sorted_peak_indices]], magnitude[peaks[top_sorted_peak_indices]], 'ro', markersize=8, label='Dominant Frequencies')
plt.title('Fourier Transform with Dominant Frequencies')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, nyquist_limit)
plt.legend()
plt.tight_layout()
plt.show()

# Spectral Shape Measures
centroid = np.sum(positive_frequencies * magnitude) / np.sum(magnitude)
bandwidth = np.sum(magnitude * (positive_frequencies - centroid)**2) / np.sum(magnitude)

print(f'Spectral Centroid: {centroid} Hz')
print(f'Spectral Bandwidth: {bandwidth} Hz')

# Spectral Flatness
flatness = np.exp(np.mean(np.log(magnitude))) / (np.mean(magnitude))
print(f'Spectral Flatness: {flatness}')

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
# Further analysis on the Fourier Transform of the input audio signal can be performed to extract more detailed information about the frequency components and their characteristics.
# This can include harmonic analysis, spectral shape measures, spectral flatness, peak detection, signal-to-noise ratio (SNR), and frequency analysis bins, among others.
# Harmonic-to-noise ratio (HNR) can be used to measure the ratio of harmonic components to noise components in the signal. A higher HNR value indicates a cleaner, more harmonic-rich signal.
	# Findings show an HNR value of 1429.41, suggesting musical or tonal elements (it's a piano key, so that makes sense) and minimal noise.
# Spectral centroid and bandwidth can be used to measure the center of mass and spread of the frequency components, respectively. A higher spectral centroid value indicates that the frequency components are more spread out, while a higher spectral bandwidth value indicates that the frequency components are more concentrated.
	# Findings show a spectral centroid (center of mass) of 521.33 Hz, suggesting that the "average" frequency is around this value and other frequencies are distributed around this value.
	# Findings show a large spectral bandwidth of 152229.43 Hz, unusually large spread/width of the frequency spectrum, which may be caused by diverse frequency components (biased to low frequencies, the left side of the FT plot)
# Spectral flatness can be used to measure the tonal quality of the signal. A higher spectral flatness value indicates that the signal is more tonal, while a lower spectral flatness value indicates that the signal is more noisy.
	# Findings show a low spectral flatness value of 0.00595 which suggests that the spectrum is more "peaky" than "flat." This indicates the presence of dominant frequency components in the signal.
# Peak detection can be used to identify the prominent frequency peaks in the signal. This can be useful for identifying specific frequency components or harmonics in the signal.
	# Dominant Frequency 1: 182.36 Hz, Amplitude: 169.18
	# Dominant Frequency 2: 177.65 Hz, Amplitude: 194.15
	# Dominant Frequency 3: 988.51 Hz, Amplitude: 250.57	* <=250 Hz
	# Dominant Frequency 4: 989.85 Hz, Amplitude: 423.54	* ~500 Hz
	# Dominant Frequency 5: 493.92 Hz, Amplitude: 2115.15	* ~2 kHz
# Signal-to-noise ratio (SNR) can be used to measure the ratio of signal power to noise power in the signal. A higher SNR value indicates a cleaner, more signal-rich signal.
	# Findings show an SNR of 31.55 dB which indicates a strong signal relative to the noise level. This is generally desirable, as it implies a clear and well-defined signal with minimal interference from noise.
# Frequency analysis bins can be used to divide the frequency spectrum into smaller frequency ranges and analyze the energy distribution within each range. This can provide insights into the frequency content and distribution of the signal.
	# The majority of the signal's energy (35k) is concentrated in the 0-1000 Hz range, indicating significant presence of low-frequency components, suggesting fundamental frequencies of musical tones.
	# In the range of 1-2 kHz, there is a lower energy level (100) indicating a reduction in signal strength compared to the lower frequency range. The drop in energy might signify a less pronounced contribution from mid-frequency components.
	# These findings align with the observations made during the Fourier Transform analysis, where dominant frequency components were identified at lower frequencies (0-250 Hz, 500 Hz, and 1 kHz). The reduced energy in the 1000-2000 Hz range may correspond to a decrease in the amplitude of these mid-frequency components.

# TASK 2: Evaluate and represent the Polar Coordinates of the Fourier Transform of the input audio signal.

# Compute the polar coordinates
polar_coordinates = np.angle(fft)

# Plot the polar coordinates using a polar plot
plt.figure(figsize=(10, 4))
plt.polar(polar_coordinates, np.abs(fft))
plt.title('Polar Coordinates of Fourier Transform')
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
