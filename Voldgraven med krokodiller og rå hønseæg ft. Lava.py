import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.fft import ifft2


def image_to_wav(image_path, wav_path):
    # Load the image
    image = plt.imread(image_path)

    # Convert image to grayscale
    if len(image.shape) == 3:  # If the image is RGB
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    # Normalize the image to range (0, 255)
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255

    # Perform inverse Fourier transform
    image_complex = ifft2(image)

    # Take the real part of the complex image
    image_real = np.real(image_complex)

    # Scale the image to range (0, 255)
    image_scaled = (image_real - np.min(image_real)) / (np.max(image_real) - np.min(image_real)) * 255

    # Convert image to bytes in range(0, 255)
    image_bytes = image_scaled.astype(np.uint8).flatten()

    # Save bytes as a WAV file
    wavfile.write(wav_path, 44100, image_bytes)

# Example usage
image_path = "spectrogram.png"
wav_path = "HubbaBubbaBirthday.wav"
image_to_wav(image_path, wav_path)

print(f"Converted image to WAV file: {wav_path}")
