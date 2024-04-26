
# Wrong image, wrong sound

import numpy as np
import matplotlib.pyplot as plt
import librosa
from PIL import Image

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
    frequency = np.fft.fftfreq(len(magnitude), 1 / sr)

    # Update frequency, magnitude, and phase to match the new length
    magnitude = magnitude[:len(magnitude) // 2]
    phase = phase[:len(phase) // 2]
    frequency = frequency[:len(frequency) // 2]

    # Convert magnitude to decibels (dB)
    magnitude_db = 20 * np.log10(magnitude)

    # Plot the Fourier Transform with raw magnitude and magnitude in dB
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

    # Adjust layout for better display
    plt.tight_layout()

    # Show the combined plot
    plt.show()

    return fft, frequency, magnitude, magnitude_db, phase


def audio_to_image(adjusted_magnitude, adjusted_phase):
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
    magnitude_image = resized_magnitude

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
    normalized_magnitude = magnitude / 255.0

    # Retrieve the original magnitude values
    magnitude = normalized_magnitude * (np.max(magnitude) - np.min(magnitude)) + np.min(magnitude)

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
    # Load the audio file
    audio_path = 'GI_GMF_B3_353_20140520_n.wav'
    y, sr = librosa.load(audio_path, sr=None)

    # Represent Input Signal
    represent_input_signal(y, sr)

    # Represent Fourier Transform
    fft, frequency, magnitude, magnitude_db, phase = represent_fourier_transform(y, sr)

    # Convert polar coordinates to image
    audio_to_image(magnitude, phase)

    # Convert image back to audio
    image_path = "Output_Image.png"

    reconstructed_audio = image_to_audio(image_path, sr)

    # wavfile.write('Reconstructed_Audio.wav', sr_original, reconstructed_audio)


if __name__ == "__main__":
    main()
