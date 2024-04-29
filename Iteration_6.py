
# Right image, right sound
# Additionally made program compatible with any audio file length

import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile
from PIL import Image
import os

def get_max_magnitude():
    # Create a sine wave alternating between -1 and 1
    sine_wave = np.tile([-1, 1], 32768)
    sine_wave = np.concatenate((sine_wave, [1]))

    # Compute the Fourier Transform of the sine wave
    fft_result = np.fft.fft(sine_wave)
    fft_result = fft_result[1:]

    # Extract the real and imaginary parts of the Fourier Transform
    real_part = np.real(fft_result)
    imag_part = np.imag(fft_result)

    # Split the real and imaginary parts into 128x256 grids
    real_grid = np.zeros((128, 256))
    imag_grid = np.zeros((128, 256))

    for k in range(128):
        real_grid[k, :] = real_part[256 * k:256 * k + 256]
        imag_grid[k, :] = imag_part[256 * k:256 * k + 256]

    # Concatenate the real and imaginary grids
    combined_grid = np.concatenate((real_grid, imag_grid), axis=0)

    # Compute the maximum magnitude
    max_magnitude = np.max(np.abs(combined_grid))

    return max_magnitude

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
    magnitude = magnitude[:len(magnitude)//2]
    phase = np.angle(fft)
    phase = phase[:len(phase) // 2]
    frequency = (sr / 2) * (np.arange(len(magnitude)) + 1) / len(magnitude)
    return fft, frequency, magnitude, phase

def resize_audio(y, desired_length):
    # Resize the audio to match the desired length
    if len(y) < desired_length:
        # Pad with zeros if the audio is shorter
        resized_audio = np.pad(y, (0, desired_length - len(y)), 'constant')
    else:
        # Trim if the audio is longer
        resized_audio = y[:desired_length]
    return resized_audio

def stretch_audio(y, desired_length):
    current_length = len(y)
    stretch_factor = desired_length / current_length

    if stretch_factor == 1:
        return y

    if stretch_factor < 1:
        # If the desired length is shorter, decimate the signal
        decimated_audio = y[::int(1/stretch_factor)]
        return resize_audio(decimated_audio, desired_length)
    else:
        # If the desired length is longer, interpolate the signal
        time_indices = np.arange(len(y))
        new_time_indices = np.linspace(0, len(y) - 1, desired_length)
        stretched_audio = np.interp(new_time_indices, time_indices, y)
        return stretched_audio

def audio_to_image(magnitude, phase, contrast_scale, MaxMag):

    # magnitude normalization (done in log scale)
    magnitude = np.log(1 + magnitude)
    magnitude = magnitude / np.log(MaxMag)

    # Apply non-linear transform to enhance visibility
    magnitude = magnitude ** contrast_scale

    # phase normalization
    phase = phase / np.pi   # [0, 2], or 0 and 2pi radians
    phase = phase + 1   # [1, 3]
    phase = phase / 2   # [0.5, 1.5] to match magnitude range

    # Convert values to pixel range [0, 65536]
    magnitude = magnitude * 65536
    phase = phase * 65536

    # Reshape magnitude and phase arrays
    magnitude_image = magnitude.reshape(128, 256)
    phase_image = phase.reshape(128, 256)

    # Combine magnitude and phase images
    combined_image = np.vstack((magnitude_image, phase_image))

    # Convert to PIL Image
    combined_image = Image.fromarray(combined_image.astype(np.uint16))

    # Save the image as PNG
    combined_image.save('Output_Image.png')

    return combined_image

def image_to_audio(sr, contrast_scale, MaxMag, original_audio_length):
    image = Image.open('Output_Image.png')
    image_array = np.array(image)
    image_array = np.double(image_array)

    # extract magnitude
    magnitude = image_array[:128, :]
    magnitude = magnitude.reshape(-1)
    magnitude = magnitude / 65536
    magnitude = magnitude ** (1 / contrast_scale)
    magnitude = magnitude * np.log(MaxMag)
    magnitude = np.exp(magnitude) - 1

    # extract phase
    phase = image_array[128:, :]
    phase = phase.reshape(-1)
    phase = phase / 65536
    phase = phase * 2
    phase = phase - 1
    phase = phase * np.pi

    # reconstruct complete amplitude and phase
    reversed_magnitude = np.flip(magnitude)
    magnitude = np.concatenate((magnitude, reversed_magnitude))
    reversed_phase = np.flip(phase)
    phase = np.concatenate((phase, -reversed_phase))

    # Combine magnitude and phase
    fft = magnitude * np.exp(1j * phase)

    # Inverse Fourier Transform
    reconstructed_audio = np.fft.ifft(fft).real

    # Stretch the reconstructed audio to match the original length
    stretched_reconstructed_audio = stretch_audio(reconstructed_audio, original_audio_length)

    # Normalize stretched reconstructed audio
    stretched_reconstructed_audio_normalized = stretched_reconstructed_audio / np.max(np.abs(stretched_reconstructed_audio))

    return stretched_reconstructed_audio_normalized

def main():
    bit16 = 65536
    contrast_scale = 0.25
    MaxMag = get_max_magnitude()

    # Load and convert the audio file to mono
    audio_path = 'GI_GMF_B3_353_20140520_n.wav'  # Specify the path to your audio file
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # Save the length of the original audio
    original_audio_length = len(y)

    # Stretch the audio to match the desired length
    resized_audio = stretch_audio(y, bit16)

    # Compute the Fourier Transform
    fft, frequency, magnitude, phase = get_fourier_transform(resized_audio, sr)

    # Save magnitude and phase as an image
    audio_to_image(magnitude, phase, contrast_scale, MaxMag)

    # Convert the image back to audio
    reconstructed_audio = image_to_audio(sr, contrast_scale, MaxMag, original_audio_length)

    # Plot the original and reconstructed audio signals
    plot_audio_signal(resized_audio, sr, 'Original Audio Signal')
    plot_audio_signal(reconstructed_audio, sr, 'Reconstructed Audio Signal')

    # Save the reconstructed audio signal
    # wavfile.write('Image_to_Audio.wav', sr, reconstructed_audio)

if __name__ == "__main__":
    main()