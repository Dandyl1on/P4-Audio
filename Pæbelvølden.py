import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile
from scipy.signal import find_peaks

def represent_fourier_transform(y, sr):
    # Compute the Fourier Transform
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    frequency = np.fft.fftfreq(len(magnitude), 1/sr)

    # Update frequency and magnitude to match the new length
    magnitude = magnitude[:len(magnitude)//2]
    frequency = frequency[:len(frequency)//2]

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

    return fft, frequency, magnitude #Ta

def evaluate_fourier_transform(fft, frequency, magnitude):
    print("\nTASK: Evaluate and represent its Fourier Transform. Do the reverse transform, save the audio, and check that it’s the same as the input audio.\n")

    # Harmonic Analysis
    nyquist_limit = len(fft) // 2
    harmonic_indices = np.where((frequency > 0) & (frequency < nyquist_limit / 2))
    noise_indices = np.where((frequency >= nyquist_limit / 2) & (frequency < nyquist_limit))

    hnr = np.sum(magnitude[harmonic_indices]) / np.sum(magnitude[noise_indices])
    print(f'Harmonic-to-Noise Ratio (HNR): {hnr}')

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

def main():
    # Load the audio file
    audio_file = 'GI_GMF_B3_353_20140520_n.wav'
    y, sr = librosa.load(audio_file)

    # Task 1.1 Represent FT, reverse FT, compare.
    fft, frequency, magnitude = represent_fourier_transform(y, sr)

    # Task 1.2 Evaluate FT.
    evaluate_fourier_transform(fft, frequency, magnitude)


if __name__ == "__main__":
    main()