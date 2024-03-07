import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile
from scipy.signal import find_peaks
from PIL import Image
from scipy.signal import butter, lfilter
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
    threshold_value = 0.075 * np.max(magnitude)
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

    # Plot the Fourier Transform with raw magnitude and magnitude in dB
    plt.figure(figsize=(12, 8))

    # Subplot 1: Raw Magnitude
    plt.subplot(2, 1, 1)
    plt.plot(frequency, magnitude)
    plt.plot(frequency[peaks], magnitude[peaks], 'ro', markersize=2, label='Dominant Frequencies')
    plt.title('Fourier Transform with Dominant Frequencies (Raw Magnitude)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()

    # Subplot 2: Magnitude in dB
    plt.subplot(2, 1, 2)
    plt.plot(frequency, magnitude_db)
    plt.plot(frequency[peaks], magnitude_db[peaks], 'ro', markersize=2, label='Dominant Frequencies')
    plt.title('Fourier Transform with Dominant Frequencies (Magnitude in dB)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()

    # Adjust layout for better display
    plt.tight_layout()

    # Show the combined plot
    plt.show()

    return fft, frequency, magnitude, magnitude_db, phase, peaks
def inverse_fourier_transform(fft_signal):
    # Perform inverse Fourier transform
    inverse_FT_transform = np.fft.ifft(fft_signal)

    return inverse_FT_transform
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
def represent_polar_coordinates(frequency, fft, phase, magnitude_scale=1.0, phase_shift=0.0):
    # Scale the magnitude and add to the phase
    scaled_magnitude = np.abs(fft)[:len(phase)] * magnitude_scale
    adjusted_phase = phase + phase_shift

    # Plot for polar coordinates
    plt.figure(figsize=(12, 8))
    plt.polar(adjusted_phase, scaled_magnitude, markersize=1)
    plt.title('Polar Coordinates of Fourier Transform in Polar Spectrum')
    plt.grid(True)
    plt.show()

    # Plot for phase information
    plt.figure(figsize=(12, 8))
    plt.plot(frequency[:len(phase)], adjusted_phase, markersize=1)
    plt.title('Phase Information')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase')
    plt.tight_layout()
    plt.show()

    return scaled_magnitude, adjusted_phase
def inverse_polar_transform(magnitude, phase):
    # Convert polar coordinates back to rectangular form
    rectangular_form = np.multiply(magnitude, np.exp(1j * phase))

    # Perform inverse Fourier transform
    inverse_PC_transform = np.fft.ifft(rectangular_form)

    return inverse_PC_transform
def audio_to_image(adjusted_magnitude, adjusted_phase, magnitude_scale=1.0):
    image_size = 256

    # Normalize the adjusted magnitude values to be in the range [0, 255]
    normalized_magnitude = ((adjusted_magnitude - np.min(adjusted_magnitude)) /
                            (np.max(adjusted_magnitude) - np.min(adjusted_magnitude)) * 255).astype(np.uint8)

    # Resize the magnitude array to the desired size
    resized_magnitude = np.resize(normalized_magnitude, (image_size // 2, image_size))

    # Resize the phase array to half of the desired size
    resized_phase = np.resize(adjusted_phase, (image_size // 2, image_size))

    # Convert polar coordinates to a square image
    polar_image = resized_phase

    # Convert magnitude to a square image
    magnitude_image = resized_magnitude * magnitude_scale  # Adjust the scaling factor

    # Create a single image by stacking magnitude on top of phase
    combined_image = np.vstack((magnitude_image, polar_image))

    # Resize the final image to the desired size
    combined_image = Image.fromarray(combined_image.astype(np.uint8)).resize((image_size, image_size))

    # Save the combined image
    combined_image.save("Output_Image.png")

def image_to_audio(image_path, sr_original):
    # Load the image
    img = Image.open(image_path)

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Split the image into magnitude and polar parts
    magnitude_img = img_array[:128, :]
    polar_img = img_array[128:, :]

    # Reshape and normalize the magnitude array
    magnitude = magnitude_img.reshape(-1)

    # Reverse the dB conversion to get raw magnitude values
    raw_magnitude = 10 ** (magnitude / 20.0)

    # Retrieve the original magnitude values
    magnitude = raw_magnitude * (np.max(raw_magnitude) - np.min(raw_magnitude)) + np.min(raw_magnitude)

    # Retrieve polar coordinates from the polar part of the image
    polar_coordinates = polar_img.reshape(-1)

    # Rescale the polar coordinates to the range [-pi, pi]
    polar_coordinates = (polar_coordinates / 255.0) * (2 * np.pi) - np.pi

    # Combine magnitude and polar coordinates into a complex array
    polar_complex = magnitude * np.exp(1j * polar_coordinates)

    # Perform inverse Fourier transform
    reconstructed_audio = np.fft.ifft(polar_complex)

    # Plot the reconstructed audio signal
    plt.figure(figsize=(10, 4))
    time = np.arange(len(reconstructed_audio)) / sr_original
    plt.plot(time, reconstructed_audio.real)
    plt.title('Reconstructed Audio Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Magnitude')
    plt.tight_layout()
    plt.show()

    return reconstructed_audio.real

def main():
    magnitude_scale = 1.0  # Default 1.0
    phase_shift = 0.0  # Default 0.0. Use radians.

    # Load the audio file without specifying the target sampling rate
    audio_file = 'GI_GMF_B3_353_20140520_n.wav'
    y, sr_original = librosa.load(audio_file, res_type='kaiser_best')

    # Apply Nyquist Theorem to determine the target sampling rate
    target_sr_multiplier = 2
    target_sr = target_sr_multiplier * sr_original

    # Reload the audio with the target sampling rate
    y, sr = librosa.load(audio_file, sr=target_sr, res_type='kaiser_best')

    # Represent Input Signal
    represent_input_signal(y, sr)

    # Represent Fourier Transform
    fft, frequency, magnitude, magnitude_db, phase, peaks = represent_fourier_transform(y, sr)

    # Evaluate Fourier Transform
    hnr, centroid, bandwidth, flatness, snr, frequency_bins, bin_energies = evaluate_fourier_transform(fft, frequency,
                                                                                                       magnitude_db, sr)

    # Inverse Fourier Transform
    inverse_FT_transform = inverse_fourier_transform(fft)

    # Save the reconstructed audio from fourier transform as a new .wav file
    wavfile.write('Inverse_FT.wav', sr_original, inverse_FT_transform.real)

    # Represent Polar Coordinates (with magnitude scale and/or phase shift)
    scaled_magnitude, adjusted_phase = represent_polar_coordinates(frequency, fft, phase, magnitude_scale, phase_shift)

    # Inverse Polar Transform with adjustable magnitude scale and/or phase shift
    inverse_PC_transform = inverse_polar_transform(magnitude, phase)

    # Save the reconstructed audio from polar coordinates as a new .wav file
    wavfile.write('Inverse_PC.wav', sr_original, inverse_PC_transform.real)

    # Convert polar coordinates to image
    audio_to_image(magnitude_db, phase+phase_shift, magnitude_scale)

    # Convert image back to audio
    image_path = "Output_Image.png"
    reconstructed_audio = image_to_audio(image_path, sr_original)

    wavfile.write('Reconstructed_Audio.wav', sr_original, reconstructed_audio)

    represent_rec_audio = represent_fourier_transform(reconstructed_audio, sr_original)

if __name__ == "__main__":
    main()
