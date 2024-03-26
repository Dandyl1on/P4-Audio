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
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    phase = np.angle(fft)
    frequency = np.fft.fftfreq(len(magnitude), 1/sr)
    return fft, frequency, magnitude, magnitude_db, phase

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
    image_size = 256

    # Normalize magnitude to [0, 255] and keep track of scaling factors
    mag_min = np.min(magnitude)
    mag_max = np.max(magnitude)
    normalized_magnitude = ((magnitude - mag_min) / (mag_max - mag_min) * 255).astype(np.uint8)

    # Normalize phase to [-π, π] and keep track of scaling factors
    phase_min = -np.pi
    phase_max = np.pi
    normalized_phase = ((phase - phase_min) / (phase_max - phase_min) * 255).astype(np.uint8)

    resized_magnitude = np.resize(normalized_magnitude, (image_size // 2, image_size))
    resized_phase = np.resize(normalized_phase, (image_size // 2, image_size))

    polar_image = resized_phase
    magnitude_image = resized_magnitude

    combined_image = np.vstack((magnitude_image, polar_image))
    combined_image = Image.fromarray(combined_image.astype(np.uint8)).resize((image_size, image_size))
    combined_image.save("Output_Image.png")

    # Return combined image and scaling factors
    return combined_image, mag_min, mag_max, phase_min, phase_max


def image_to_audio(combined_image, mag_min, mag_max, phase_min, phase_max, image_size=256):
    # Resize the combined image to its original dimensions
    combined_image = combined_image.resize((image_size, image_size))

    # Split the combined image into magnitude and phase parts
    magnitude_image = np.array(combined_image.crop((0, 0, image_size, image_size // 2)))
    polar_image = np.array(combined_image.crop((0, image_size // 2, image_size, image_size)))

    # Resize magnitude and phase images to original sizes
    resized_magnitude = np.resize(magnitude_image, (image_size // 2, image_size))
    resized_phase = np.resize(polar_image, (image_size // 2, image_size))

    # Rescale magnitude and phase back to original ranges
    recon_mag = ((resized_magnitude / 255) * (mag_max - mag_min) + mag_min).astype(np.float32)
    recon_phase = ((resized_phase / 255) * (phase_max - phase_min) + phase_min).astype(np.float32)

    return recon_mag, recon_phase

def reconstruct_audio(recon_mag, recon_phase):

    # Combine magnitude and phase to obtain the complex spectrum
    complex_spectrum = recon_mag * np.exp(1j * recon_phase)

    # Perform the inverse Fourier transform
    reconstructed_audio = np.fft.ifft(complex_spectrum).real

    return reconstructed_audio.real

def main():
    # Load the audio file
    audio_path = 'GI_GMF_B3_353_20140520_n.wav'
    y, sr = librosa.load(audio_path, sr=None)

    # Plot the audio signal
    plot_audio_signal(y, sr, 'Original Audio Signal')

    # Compute the Fourier Transform
    fft, frequency, magnitude, magnitude_db, phase = get_fourier_transform(y, sr)

    # Plot the Fourier Transform
    plot_fourier_transform(frequency, magnitude)

    # Inverse Fourier Transform
    inverse_FT_transform = inverse_fourier_transform(fft)

    # Plot the phase spectrum
    plot_phase(frequency, phase)

    # Convert the audio signal to an image
    image, mag_min, mag_max, phase_min, phase_max = audio_to_image(magnitude_db, phase)

    # Convert the image back to audio
    recon_mag, recon_phase = image_to_audio(image, mag_min, mag_max, phase_min, phase_max)

    reconstructed_audio_signal = reconstruct_audio(recon_mag, recon_phase)

    plot_audio_signal(reconstructed_audio_signal, sr, 'Reconstructed Audio Signal')

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
