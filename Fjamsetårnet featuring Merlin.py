import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile
from PIL import Image

def plot_audio_signal(y, sr, name):
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(y)) / sr, y)
    plt.title(name)
    plt.xlabel('Time (s)')
    plt.ylabel('Magnitude')
    plt.tight_layout()
    plt.show()

def get_fourier_transform(y, sr):
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    phase = np.angle(fft)
    frequency = np.fft.fftfreq(len(magnitude), 1/sr)
    return fft, frequency, magnitude, phase

def plot_fourier_transform(frequency, magnitude):
    plt.figure(figsize=(12, 8))
    plt.plot(frequency, magnitude)
    plt.title('Fourier Transform')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.tight_layout()
    plt.show()

def inverse_fourier_transform(fft_signal):
    inverse_FT_transform = np.fft.ifft(fft_signal)
    return inverse_FT_transform.real

def plot_phase(frequency, phase):
    plt.figure(figsize=(10, 4))
    plt.plot(frequency, phase)
    plt.title('Phase Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase')
    plt.tight_layout()
    plt.show()

# 4	Wrong image (2 row), wrong sound (decibel, right ranges)

def audio_to_image(magnitude, phase):
    # Convert magnitude to decibels
    # magnitude = 20 * np.log10(magnitude)

    # Normalize magnitude to [0, 255]
    normalized_magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude)) * 255

    # Scale phase to [0, 255]
    normalized_phase = (phase + np.pi) / (2 * np.pi) * 255

    # Reshape magnitude and phase arrays
    magnitude_image = normalized_magnitude.reshape((256, -1))[:128]  # Reshape to have 256 columns
    phase_image = normalized_phase.reshape((256, -1))[:128] # Reshape to have 256 columns

    # Combine magnitude and phase images
    combined_image = np.vstack((magnitude_image, phase_image))

    # Convert to PIL Image
    combined_image = Image.fromarray(combined_image.astype(np.uint8))

    # Save the image (optional)
    combined_image.save('Output_Image.png')

    # Print debugging information
    print("Magnitude (dB) min:", np.min(magnitude))
    print("Magnitude (dB) max:", np.max(magnitude))

    return combined_image

def image_to_audio(image, sr):
    # Convert image to numpy array
    image_array = np.array(image)

    # Reshape the image array back to separate magnitude and phase
    magnitude_rows = image_array[:image_array.shape[0] // 2]
    phase_rows = image_array[image_array.shape[0] // 2:]

    # Reshape magnitude and phase arrays
    magnitude = magnitude_rows.reshape(-1)
    phase = phase_rows.reshape(-1)

    # Convert decibel values back to linear scale for magnitude
    # magnitude = 10 ** (magnitude / 20)

    # Ensure magnitude values are within a reasonable range
    magnitude = np.clip(magnitude, 1e-6, None)

    # Scale phase back to [-π, π]
    phase = (phase / 255) * 2 * np.pi - np.pi

    # Combine magnitude and phase
    fft = magnitude * np.exp(1j * phase)

    # Inverse Fourier Transform
    reconstructed_audio = np.fft.ifft(fft).real

    # Normalize reconstructed audio
    reconstructed_audio_normalized = reconstructed_audio / np.max(np.abs(reconstructed_audio))

    # Print debugging information
    print("Reconstructed audio min:", np.min(reconstructed_audio_normalized))
    print("Reconstructed audio max:", np.max(reconstructed_audio_normalized))

    return reconstructed_audio_normalized

def main():
    # Load the audio file
    audio_path = 'GI_GMF_B3_353_20140520_n.wav'
    y, sr = librosa.load(audio_path, sr=None)

    # Apply nyquist theorem
    sr = sr // 2

    # Plot the audio signal
    plot_audio_signal(y, sr, 'Original Audio Signal')

    # Compute the Fourier Transform
    fft, frequency, magnitude, phase = get_fourier_transform(y, sr)

    # Plot the Fourier Transform
    plot_fourier_transform(frequency, magnitude)

    # Inverse Fourier Transform
    inverse_FT_transform = inverse_fourier_transform(fft)

    # Plot the phase spectrum
    plot_phase(frequency, phase)

    # Convert the audio signal to an image
    image = audio_to_image(magnitude, phase)

    # Convert the image back to audio
    reconstructed_audio = image_to_audio(image, sr)

    # Plot the reconstructed audio signal
    plot_audio_signal(reconstructed_audio, sr, 'Reconstructed Audio Signal')

    # Save the reconstructed audio signal
    wavfile.write('Image_to_Audio.wav', sr, reconstructed_audio)

if __name__ == "__main__":
    main()
