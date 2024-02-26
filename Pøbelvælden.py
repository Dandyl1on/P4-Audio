import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile
from scipy.signal import find_peaks

# Load the audio file
audio_file = 'GI_GMF_B3_353_20140520_n.wav'
y, sr = librosa.load(audio_file)

print(sr)

# TASK 1: Evaluate and represent the Fourier Transform of the input audio signal.

# Compute the Fourier Transform
fft = np.fft.fft(y)
magnitude = np.abs(fft)
frequency = np.fft.fftfreq(len(magnitude), 1/sr)  # Updated frequency calculation

magnitude = magnitude[:len(magnitude)//2]
frequency = frequency[:len(frequency)//2]  # Update frequency to match the new magnitude length

# Plot the original audio signal
plt.figure(figsize=(10, 4))
plt.plot(np.arange(len(y)) / sr, y)
plt.title('Original Audio Signal')
plt.xlabel('Time (s)')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()

# Set Nyquist frequency limit (for visualization purposes only)
nyquist_limit = sr / 2

# Plot the Fourier Transform
plt.figure(figsize=(10, 4))
plt.plot(frequency, magnitude)
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
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()

# Trying methods for analysis and evaluation of the fourier transform of the audio signal.

print("\nTASK: Evaluate and represent its Fourier Transform. Do the reverse transform, save the audio, and check that it’s the same as the input audio.\n")

# Harmonic Analysis
harmonic_indices = np.where((frequency > 0) & (frequency < nyquist_limit / 2))
noise_indices = np.where((frequency >= nyquist_limit / 2) & (frequency < nyquist_limit))

hnr = np.sum(magnitude[harmonic_indices]) / np.sum(magnitude[noise_indices])
print(f'Harmonic-to-Noise Ratio (HNR): {hnr}')

# Find peaks in the magnitude spectrum with a height threshold
# Here, we set the threshold to be 5% of the maximum magnitude
threshold_value = 0.05 * np.max(magnitude)
peaks, _ = find_peaks(magnitude, height=threshold_value)

# Sort peaks by magnitude in ascending order
sorted_peak_indices = np.argsort(magnitude[peaks])

# Select the top 5 peaks
top_sorted_peak_indices = sorted_peak_indices[-5:]

# Print the top 5 dominant frequencies and their magnitudes (from lowest to highest)
top_frequencies = frequency[peaks[top_sorted_peak_indices]]
top_magnitudes = magnitude[peaks[top_sorted_peak_indices]]

# Combine frequencies and magnitudes into a list of tuples
dominant_info = list(zip(top_frequencies, top_magnitudes))

# Sort the list by magnitude in ascending order
dominant_info.sort(key=lambda x: x[1])

# Print the sorted dominant frequencies and their magnitudes
for freq, mag in dominant_info:
    print(f'Dominant Frequency: {freq} Hz, Magnitude: {mag}')

# Plot the identified peaks on the Fourier Transform plot
plt.figure(figsize=(10, 4))
plt.plot(frequency, magnitude)
plt.plot(frequency[peaks[top_sorted_peak_indices]], magnitude[peaks[top_sorted_peak_indices]], 'ro', markersize=8, label='Dominant Frequencies')
plt.title('Fourier Transform with Dominant Frequencies')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, nyquist_limit)
plt.legend()
plt.tight_layout()
plt.show()

# Spectral Shape Measures
centroid = np.sum(frequency * magnitude) / np.sum(magnitude)
bandwidth = np.sum(magnitude * (frequency - centroid)**2) / np.sum(magnitude)

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
bin_indices = np.digitize(frequency, frequency_bins)
bin_energies = [np.sum(magnitude[bin_indices == i]) for i in range(1, len(frequency_bins))]

# Plot the Frequency Analysis Bins
plt.figure(figsize=(10, 4))
plt.bar(frequency_bins[:-1], bin_energies, width=np.diff(frequency_bins), align='edge')
plt.title('Frequency Analysis Bins')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Energy')
plt.tight_layout()
plt.show()

# TASK 2: Evaluate and represent the Polar Coordinates of the Fourier Transform of the input audio signal.

# Compute the polar coordinates using the entire frequency range
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
plt.title('Reconstructed Audio Signal from Fourier Transform')
plt.xlabel('Time (s)')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()

print("\nTASK: Evaluate and represent the polar coordinates. Do the reverse transform, save the audio, and check that it’s the same as the input audio.\n")

# Find peaks in polar coordinates with a height threshold
polar_peaks, _ = find_peaks(magnitude, height=threshold_value)

# Sort polar peaks by magnitude in ascending order
sorted_polar_indices = np.argsort(magnitude[polar_peaks])

# Select the top 5 polar peaks
top_sorted_polar_indices = sorted_polar_indices[-5:]

# Ensure that the indices are within bounds
top_sorted_polar_indices = top_sorted_polar_indices[top_sorted_polar_indices < len(polar_peaks)]

# Convert radian angles to degrees
polar_angles_degrees = np.degrees(polar_coordinates[polar_peaks[top_sorted_polar_indices]])

# Extract top 5 dominant frequencies and their magnitudes
top_polar_frequencies = frequency[polar_peaks[top_sorted_polar_indices]]
top_polar_magnitudes = magnitude[polar_peaks[top_sorted_polar_indices]]

# Combine frequencies, magnitudes, and angles into a list of tuples
top_polar_info = list(zip(top_polar_frequencies, top_polar_magnitudes, polar_coordinates[polar_peaks[top_sorted_polar_indices]], polar_angles_degrees))

# Sort the list by magnitude in ascending order
top_polar_info.sort(key=lambda x: x[1])

# Print the sorted top 5 dominant frequencies, magnitudes, and angles in both radians and degrees for reference
for freq, mag, angle_rad, angle_deg in top_polar_info:
    print(f'Dominant Frequency: {freq} Hz, Magnitude: {mag}, Angle (Radians): {angle_rad}, Angle (Degrees): {angle_deg} degrees')

# Plot the identified polar peaks on the Polar Coordinates plot
plt.figure(figsize=(10, 4))
plt.polar(polar_coordinates, np.abs(fft))
plt.plot(polar_coordinates[polar_peaks[top_sorted_polar_indices]], magnitude[polar_peaks[top_sorted_polar_indices]], 'ro', markersize=8, label='Dominant Frequencies')
plt.title('Polar Coordinates of Fourier Transform with Dominant Frequencies')
plt.tight_layout()
plt.legend()
plt.show()

# TASK 3: Convert the polar coordinates of the Fourier transform to square images. Save the image, reload it, do the inverse transform, and check that it’s the same as the input audio.

# Convert polar coordinates to a square image
image_size = int(np.sqrt(len(polar_coordinates)))
polar_image = polar_coordinates[:image_size**2].reshape((image_size, image_size))

# Plot the grayscale image of polar coordinates
plt.figure(figsize=(8, 8))
plt.imshow(polar_image, cmap='gray')  # Use 'gray' colormap for grayscale
plt.title('Grayscale Image of Polar Coordinates')
plt.colorbar()
plt.tight_layout()
plt.show()

# Save the polar image as a PNG file
polar_image_path = 'polar_image.png'
plt.imsave(polar_image_path, polar_image, cmap='gray')

# Reload the polar image
reloaded_polar_image = plt.imread(polar_image_path)

# Plot the reloaded grayscale image of polar coordinates
plt.figure(figsize=(8, 8))
plt.imshow(reloaded_polar_image, cmap='gray')
plt.title('Reloaded Grayscale Image of Polar Coordinates')
plt.colorbar()
plt.tight_layout()
plt.show()

# Inverse Fourier Transform from the reloaded polar image
reconstructed_signal = np.fft.ifft(reloaded_polar_image.flatten()).real

# Plot the Reconstructed Audio Signal from the reloaded polar image
plt.figure(figsize=(10, 4))
plt.plot(np.arange(len(reconstructed_signal)) / sr, reconstructed_signal)
plt.title('Reconstructed Audio Signal from Reloaded Polar Image')
plt.xlabel('Time (s)')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()

# Convert polar coordinates to a square image for the reconstructed signal
reconstructed_polar_coordinates = np.angle(np.fft.fft(reconstructed_signal))
reconstructed_polar_image = reconstructed_polar_coordinates[:image_size**2].reshape((image_size, image_size))

# Plot the grayscale image of polar coordinates for the reconstructed signal
plt.figure(figsize=(8, 8))
plt.imshow(reconstructed_polar_image, cmap='gray')
plt.title('Grayscale Image of Reconstructed Polar Coordinates')
plt.colorbar()
plt.tight_layout()
plt.show()