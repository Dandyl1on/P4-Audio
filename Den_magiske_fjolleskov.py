import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile
from PIL import Image
from scipy.ndimage import gaussian_filter1d

# Function to represent the input audio signal
def represent_input_signal(y, sr):
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(y)) / sr, y)
    plt.title('Original Audio Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Magnitude')
    plt.tight_layout()
    plt.show()

# Function to represent the Fourier Transform and its analysis
def represent_fourier_transform(y, sr):
    # Compute the Fourier Transform
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    phase = np.angle(fft)
    frequency = np.fft.fftfreq(len(magnitude), 1/sr)

    # Update frequency, magnitude, and phase
    magnitude, phase, frequency = magnitude[:len(magnitude)//2], phase[:len(phase)//2], frequency[:len(frequency)//2]

    # Convert magnitude to decibels (dB)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)  # Adding a small value to avoid log(0)

    # Plot the Fourier Transform
    plot_fourier_transform(frequency, magnitude, magnitude_db)

    return fft, frequency, magnitude, magnitude_db, phase

# Function to plot Fourier Transform
def plot_fourier_transform(frequency, magnitude, magnitude_db):
    plt.figure(figsize=(12, 8))

    # Subplot 1: Raw Magnitude
    plt.subplot(2, 1, 1)
    plt.plot(frequency, magnitude)
    plt.title('Fourier Transform (Raw Magnitude)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')

    # Subplot 2: Magnitude in dB
    plt.subplot(2, 1, 2)
    plt.plot(frequency, magnitude_db)
    plt.title('Fourier Transform (Magnitude in dB)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')

    plt.tight_layout()
    plt.show()

# Function for inverse Fourier Transform
def inverse_fourier_transform(fft_signal):
    inverse_FT_transform = np.fft.ifft(fft_signal)
    return inverse_FT_transform

# Function to represent polar coordinates
def represent_polar_coordinates(frequency, fft, phase, magnitude_scale=1.0, phase_shift=0.0):
    scaled_magnitude = np.abs(fft)[:len(phase)] * magnitude_scale
    adjusted_phase = phase

    plot_polar_coordinates(adjusted_phase, scaled_magnitude)
    plot_phase_information(frequency[:len(phase)], adjusted_phase)

    return scaled_magnitude, adjusted_phase

# Function to plot polar coordinates
def plot_polar_coordinates(adjusted_phase, scaled_magnitude):
    plt.figure(figsize=(12, 8))
    plt.polar(adjusted_phase, scaled_magnitude, markersize=1)
    plt.title('Polar Coordinates of Fourier Transform in Polar Spectrum')
    plt.grid(True)
    plt.show()

# Function to plot phase information
def plot_phase_information(frequency, adjusted_phase):
    plt.figure(figsize=(12, 8))
    plt.plot(frequency, adjusted_phase, markersize=1)
    plt.title('Phase Information')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase')
    plt.tight_layout()
    plt.show()

# Function for inverse polar transform
def inverse_polar_transform(magnitude, phase):
    rectangular_form = np.multiply(magnitude, np.exp(1j * phase))
    inverse_PC_transform = np.fft.ifft(rectangular_form)
    return inverse_PC_transform

# Function to convert audio to image
def audio_to_image(adjusted_magnitude, adjusted_phase, magnitude_scale=1.0):
    image_size = 256

    # Normalize magnitude to [0, 255]
    normalized_magnitude = ((adjusted_magnitude - np.min(adjusted_magnitude)) /
                            (np.max(adjusted_magnitude) - np.min(adjusted_magnitude)) * 255).astype(np.uint8)

    # Normalize phase to [-π, π]
    normalized_phase = adjusted_phase - np.pi

    # Scale phase to [0, 255]
    normalized_phase = ((normalized_phase - np.min(normalized_phase)) /
                        (np.max(normalized_phase) - np.min(normalized_phase)) * 255).astype(np.uint8)

    resized_magnitude = np.resize(normalized_magnitude, (image_size // 2, image_size))
    resized_phase = np.resize(normalized_phase, (image_size // 2, image_size))

    polar_image = resized_phase
    magnitude_image = resized_magnitude * magnitude_scale

    combined_image = np.vstack((magnitude_image, polar_image))
    combined_image = Image.fromarray(combined_image.astype(np.uint8)).resize((image_size, image_size))
    combined_image.save("Output_Image.png")

    return combined_image



def image_to_audio(image, sr):
    img_array = np.array(image)

    magnitude_img = img_array[:128, :]
    polar_img = img_array[128:, :]

    # Reverse normalization of magnitude
    magnitude = magnitude_img.reshape(-1)
    magnitude = (magnitude / 255.0) * (np.max(magnitude) - np.min(magnitude)) + np.min(magnitude)
    raw_magnitude = 10 ** ((magnitude - np.max(magnitude) + 73) / 20.0)  # Adjust the +6 for volume increase

    # Reverse normalization of polar coordinates
    polar_coordinates = polar_img.reshape(-1)
    polar_coordinates = ((polar_coordinates + 100) / 255.0) * (2 * np.pi) - np.pi

    # Reverse phase bias
    bias = np.mean(polar_coordinates)
    polar_coordinates += bias

    # Reconstruct the complex Fourier coefficients
    polar_complex = raw_magnitude * np.exp(1j * polar_coordinates)

    # Perform inverse Fourier transform
    reconstructed_audio = np.fft.ifft(polar_complex)

    # Plot reconstructed audio
    plot_reconstructed_audio(reconstructed_audio, sr)

    return reconstructed_audio.real


def plot_reconstructed_audio(reconstructed_audio, sr):
    plt.figure(figsize=(10, 4))
    time = np.arange(len(reconstructed_audio)) / sr
    plt.plot(time, reconstructed_audio.real)
    plt.title('Reconstructed Audio Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Magnitude')
    plt.tight_layout()
    plt.show()

# Function to perform the main operations
def main():
    magnitude_scale = 1.0
    phase_shift = 0

    audio_file = 'GI_GMF_B3_353_20140520_n.wav'
    y, sr = librosa.load(audio_file, sr=44100)

    represent_input_signal(y, sr)
    fft, frequency, magnitude, magnitude_db, phase = represent_fourier_transform(y, sr)
    inverse_FT_transform = inverse_fourier_transform(fft)
    wavfile.write('Inverse_FT.wav', sr, inverse_FT_transform.real)

    represent_polar_coordinates(frequency, fft, phase, magnitude_scale, phase_shift)
    inverse_PC_transform = inverse_polar_transform(magnitude, phase)
    wavfile.write('Inverse_PC.wav', sr, inverse_PC_transform.real)

    audio_to_image(magnitude_db, phase + phase_shift, magnitude_scale)

    image = audio_to_image(magnitude_db, phase + phase_shift, magnitude_scale)

    reconstructed_audio_signal = image_to_audio(image, sr)

    fft1, frequency1, magnitude1, magnitude_db1, phase1 = represent_fourier_transform(reconstructed_audio_signal, sr)

    wavfile.write('Reconstructed_Audio.wav', sr, reconstructed_audio_signal)

    represent_polar_coordinates(frequency1, fft1, phase1, magnitude_scale, phase_shift)
    plot_polar_coordinates(phase1, magnitude1)

if __name__ == "__main__":
    main()
