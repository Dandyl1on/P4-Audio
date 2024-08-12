
# Wrong image, wrong sound
# Unnormalized top and bot


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

def represent_fourier_transform(y, sr):
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    phase = np.angle(fft)
    frequency = np.fft.fftfreq(len(magnitude), 1 / sr)

    magnitude = magnitude[:len(magnitude) // 2]
    phase = phase[:len(phase) // 2]
    frequency = frequency[:len(frequency) // 2]

    magnitude_db = 20 * np.log10(magnitude)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(frequency, magnitude)
    plt.title('Fourier Transform (Raw Magnitude)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')

    # Subplot 2:  in dB
    plt.subplot(2, 1, 2)
    plt.plot(frequency, magnitude_db)
    plt.title('Fourier Transform (Magnitude in dB)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')

    plt.tight_layout()
    plt.show()

    return fft, frequency, magnitude, magnitude_db, phase

def audio_to_image(magnitude, phase):
    image_size = 256

    normalized_magnitude = ((magnitude - np.min(magnitude)) /
                            (np.max(magnitude) - np.min(magnitude)) * 255).astype(np.uint8)

    resized_magnitude = np.resize(normalized_magnitude, (image_size // 2, image_size))

    resized_phase = np.resize(phase, (image_size // 2, image_size))

    polar_image = resized_phase

    magnitude_image = resized_magnitude

    combined_image = np.vstack((magnitude_image, polar_image))

    combined_image = Image.fromarray(combined_image.astype(np.uint8)).resize((image_size, image_size))

    combined_image.save("Output_Image.png")

def image_to_audio(image_path, sr):
    img = Image.open(image_path)

    img_array = np.array(img)

    magnitude_img = img_array[:128, :]
    polar_img = img_array[128:, :]

    magnitude = magnitude_img.reshape(-1)
    normalized_magnitude = magnitude / 255.0

    magnitude = normalized_magnitude * (np.max(magnitude) - np.min(magnitude)) + np.min(magnitude)

    polar_coordinates = polar_img.reshape(-1)

    polar_coordinates = (polar_coordinates / 255.0) * (2 * np.pi) - np.pi

    polar_complex = magnitude * np.exp(1j * polar_coordinates)

    reconstructed_audio = np.fft.ifft(polar_complex)

    plt.figure(figsize=(10, 4))
    time = np.arange(len(reconstructed_audio)) / sr
    plt.plot(time, reconstructed_audio.real)
    plt.title('Reconstructed Audio Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Magnitude')
    plt.tight_layout()
    plt.show()

    return reconstructed_audio.real

def main():
    audio_path = 'GI_GMF_B3_353_20140520_n.wav'
    y, sr = librosa.load(audio_path, sr=None)
    plot_audio_signal(y, sr, "Original Audio Signal")
    fft, frequency, magnitude, magnitude_db, phase = represent_fourier_transform(y, sr)
    audio_to_image(magnitude, phase)
    image_path = "Output_Image.png"
    reconstructed_audio = image_to_audio(image_path, sr)
    wavfile.write('Reconstructed_Audio.wav', sr, reconstructed_audio)
if __name__ == "__main__":
    main()
