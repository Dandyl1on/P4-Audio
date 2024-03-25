import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile
from PIL import Image

def plot_audio_signal(y, sr):
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(y)) / sr, y)
    plt.title('Original Audio Signal')
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

def audio_to_image(magnitude, phase):
    # Image size
    image_size = len(magnitude)

    # Normalize magnitude to [0, 255]
    normalized_magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude)) * 255

    # Scale phase to [0, 255]
    normalized_phase = (phase + np.pi) / (2 * np.pi) * 255

    # Create magnitude and phase images
    magnitude_image = normalized_magnitude.reshape((1, -1))
    phase_image = normalized_phase.reshape((1, -1))

    # Stack magnitude and phase images
    combined_image = np.vstack((magnitude_image, phase_image))

    # Convert to PIL Image
    combined_image = Image.fromarray(combined_image.astype(np.uint8))

    return combined_image

def image_to_audio(image, sr):
    # Convert image to numpy array
    image_array = np.array(image)

    # Split magnitude and phase
    magnitude = image_array[0]
    phase = image_array[1]

    # Scale magnitude back to original range
    magnitude = (magnitude / 255) * (np.max(magnitude) - np.min(magnitude)) + np.min(magnitude)

    # Scale phase back to [-π, π]
    phase = (phase / 255) * 2 * np.pi - np.pi

    # Combine magnitude and phase
    fft = (magnitude*(6*np.pi)) * np.exp(1j * phase) # Multiply by 6*np.pi to increase the magnitude when converting back to audio because the magnitude was scaled down to [0, 1] during image conversion

    # Inverse Fourier Transform
    reconstructed_audio = np.fft.ifft(fft).real

    return reconstructed_audio

def main():
    # Load the audio file
    audio_path = 'GI_GMF_B3_353_20140520_n.wav'
    y, sr = librosa.load(audio_path, sr=None)

    # Plot the audio signal
    plot_audio_signal(y, sr)

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
    plot_audio_signal(reconstructed_audio, sr)

    # Save the reconstructed audio signal
    wavfile.write('Image_to_Audio.wav', sr, reconstructed_audio)

if __name__ == "__main__":
    main()
