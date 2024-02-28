import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile
from scipy.signal import find_peaks

def represent_input_signal(y, sr):
    # Plot the original audio signal
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(y)) / sr, y)
    plt.title('Original Audio Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Magnitude')
    plt.tight_layout()
    plt.show()

def represent_fourier_transform(y, sr):
    # Compute the Fourier Transform
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    phase = np.angle(fft)  # Extract phase information
    frequency = np.fft.fftfreq(len(magnitude), 1/sr)

    # Update frequency, magnitude, and phase to match the new length
    magnitude = magnitude[:len(magnitude)//2]
    phase = phase[:len(phase)//2]
    frequency = frequency[:len(frequency)//2]

    # Convert magnitude to decibels (dB)
    magnitude_db = 20 * np.log10(magnitude)

    # Find peaks in the magnitude spectrum with a height threshold
    threshold_value = 0.05 * np.max(magnitude)
    peaks, _ = find_peaks(magnitude, height=threshold_value)

    # Sort peaks by magnitude in ascending order
    sorted_peak_indices = np.argsort(magnitude[peaks])

    # Select the top 5 peaks
    top_sorted_peak_indices = sorted_peak_indices[-5:]

    # Print the top 5 dominant frequencies and their magnitudes (from lowest to highest)
    top_frequencies = frequency[peaks[top_sorted_peak_indices]]
    top_magnitudes = magnitude[peaks[top_sorted_peak_indices]]
    top_magnitudes_db = magnitude_db[peaks[top_sorted_peak_indices]]

    # Combine frequencies and magnitudes into a list of tuples
    dominant_info = list(zip(top_frequencies, top_magnitudes, top_magnitudes_db))

    # Sort the list by frequency in ascending order
    dominant_info.sort(key=lambda x: x[0])

    # Print the sorted dominant frequencies, raw magnitudes, and magnitudes in dB
    for freq, mag, mag_db in dominant_info:
        print(f'Dominant Frequency: {freq} Hz, Magnitude: {mag}, Magnitude (dB): {mag_db}')

    # Plot the Fourier Transform, its magnitude in dB, and phase with dominant frequencies highlighted
    plt.figure(figsize=(12, 8))

    # Subplot for raw magnitude
    plt.subplot(3, 1, 1)
    plt.plot(frequency, magnitude)
    plt.plot(frequency[peaks], magnitude[peaks], 'ro', markersize=2, label='Dominant Frequencies')
    plt.title('Fourier Transform with Dominant Frequencies (Raw Magnitude)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()

    # Subplot for magnitude in dB
    plt.subplot(3, 1, 2)
    plt.plot(frequency, magnitude_db)
    plt.plot(frequency[peaks], magnitude_db[peaks], 'ro', markersize=2, label='Dominant Frequencies')
    plt.title('Fourier Transform with Dominant Frequencies (Magnitude in dB)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()

    # Subplot for phase
    plt.subplot(3, 1, 3)
    plt.plot(frequency, phase)
    plt.title('Phase Information')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase')
    plt.tight_layout()
    plt.show()

    return fft, frequency, magnitude_db, phase

def evaluate_fourier_transform(fft, frequency, magnitude_db, sr, num_bins=10):
    nyquist_limit = sr / 2
    harmonic_indices = np.where((frequency > 0) & (frequency < nyquist_limit / 2))
    noise_indices = np.where((frequency >= nyquist_limit / 2) & (frequency < nyquist_limit))
    hnr = np.sum(np.abs(fft)[harmonic_indices]) / np.sum(np.abs(fft)[noise_indices])
    print(f'\nHarmonic-to-Noise Ratio (HNR): {hnr:.2f} (High indicates a clearer signal))')

    signal_power = np.sum(np.abs(fft)[harmonic_indices])
    noise_power = np.sum(np.abs(fft)[noise_indices])
    snr = 10 * np.log10(signal_power / noise_power)
    print(f'\nSignal-to-Noise Ratio (SNR): {snr:.2f} dB (High indicates a clearer signal))')

    centroid = np.sum(frequency * np.abs(fft)[:len(frequency)]) / np.sum(np.abs(fft)[:len(frequency)])
    bandwidth = np.sum(np.abs(fft)[:len(frequency)] * (frequency - centroid) ** 2) / np.sum(
        np.abs(fft)[:len(frequency)])
    print(
        f'\nSpectral Centroid: {centroid:.2f} Hz (Energy distributed around this frequency(center of mass/average frequency))')
    print(f'\nSpectral Bandwidth: {bandwidth:.2f} Hz (High indicates a broader frequency range)')

    flatness = np.exp(np.mean(np.log(np.abs(fft)[:len(frequency)]))) / (np.mean(np.abs(fft)[:len(frequency)]))
    print(f'\nSpectral Flatness: {flatness:.2f} (High indicates a more flat spectrum)')

    frequency_bins = np.linspace(0, nyquist_limit, num_bins)  # Adjust the number of bins as needed
    bin_indices = np.digitize(frequency, frequency_bins)
    bin_energies = [np.sum(np.abs(fft)[:len(frequency)][bin_indices == i]) for i in range(1, len(frequency_bins))]

    # Plot the Frequency Analysis Bins
    plt.figure(figsize=(10, 4))
    plt.bar(frequency_bins[:-1], bin_energies, width=np.diff(frequency_bins), align='edge')
    plt.title('Frequency Analysis Bins')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Energy')
    plt.tight_layout()
    plt.show()

    return hnr, centroid, bandwidth, flatness, snr, frequency_bins, bin_energies

def represent_polar_coordinates(frequency, fft, phase):
    # Plot the polar coordinates in the frequency domain with lines connecting points
    plt.figure(figsize=(8, 6))
    plt.plot(frequency, phase, 'b-')
    plt.title('Polar Coordinates of Fourier Transform in Frequency Domain')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase')
    plt.grid(True)
    plt.show()


def main():
    # Load the audio file
    audio_file = 'GI_GMF_B3_353_20140520_n.wav'
    y, sr = librosa.load(audio_file)

    # Represent Input Signal
    represent_input_signal(y, sr)

    # Represent Fourier Transform
    fft, frequency, magnitude_db, phase = represent_fourier_transform(y, sr)

    # Evaluate Fourier Transform
    hnr, centroid, bandwidth, flatness, snr, frequency_bins, bin_energies = evaluate_fourier_transform(fft, frequency, magnitude_db, sr)

    # Represent Polar Coordinates
    represent_polar_coordinates(frequency, fft, phase)

if __name__ == "__main__":
    main()
