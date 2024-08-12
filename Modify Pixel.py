import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile
from PIL import Image
import math

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

def audio_to_image(magnitude, phase, contrast_scale, MaxMag):
    # magnitude normalization (done in log scale)
    magnitude = np.log(1 + magnitude)
    magnitude = magnitude / np.log(MaxMag)

    # apply square root transformation for contrast
    magnitude = magnitude ** contrast_scale

    # phase normalization
    phase = phase / np.pi
    phase = phase + 1
    phase = phase / 2

    # scale to 65536 for image representation
    magnitude = magnitude * 65536
    phase = phase * 65536

    # reshape magnitude and phase arrays
    magnitude_image = magnitude.reshape(128, 256)
    phase_image = phase.reshape(128, 256)

    # combine magnitude and phase images
    combined_image = np.vstack((magnitude_image, phase_image))

    # convert to PIL image
    combined_image = Image.fromarray(combined_image.astype(np.uint16))

    # save the image
    combined_image.save('Output_Image.png')

    return combined_image

def image_to_audio(sr, contrast_scale, MaxMag, factor):
    # Load the original image
    original_image = Image.open('Output_Image.png')
    original_image_array = np.array(original_image)
    original_image_array = np.double(original_image_array)

    start_row, start_pixel = 128, 0

    while start_row < 256:
        # Reload the original image for each iteration
        image_array = original_image_array.copy()

        for i in range(30):
            row = start_row
            col = start_pixel + i
            if col >= 256:
                break
            image_array[row, col] *= factor

        # Save the modified image with start_row and start_pixel in the name
        filename = f'Modified_Image_row{start_row}_pixel{start_pixel}.png'
        modified_image = Image.fromarray(image_array.astype(np.uint16))
        modified_image.save(filename)

        # Extract magnitude
        magnitude = image_array[:128, :].reshape(-1)
        magnitude = magnitude / 65536
        magnitude = magnitude ** (1 / contrast_scale)
        magnitude = magnitude * np.log(MaxMag)
        magnitude = np.exp(magnitude) - 1

        # Extract phase
        phase = image_array[128:, :].reshape(-1)
        phase = phase / 65536
        phase = phase * 2
        phase = phase - 1
        phase = phase * np.pi

        # Reconstruct complete amplitude and phase
        reversed_magnitude = np.flip(magnitude)
        magnitude = np.concatenate((magnitude, reversed_magnitude))
        reversed_phase = np.flip(phase)
        phase = np.concatenate((phase, -reversed_phase))

        # Combine magnitude and phase
        fft = magnitude * np.exp(1j * phase)

        # Inverse Fourier Transform
        reconstructed_audio = np.fft.ifft(fft).real

        # Normalize reconstructed audio
        reconstructed_audio_normalized = reconstructed_audio / np.max(np.abs(reconstructed_audio))

        # Save the reconstructed audio signal with start_row and start_pixel in the name
        audio_filename = f'Reconstructed_Audio_row{start_row}_pixel{start_pixel}.wav'
        wavfile.write(audio_filename, sr, (reconstructed_audio_normalized * 32767).astype(np.int16))

        # Update start_pixel and start_row
        start_pixel += 30
        if start_pixel >= 256:
            start_row += 1
            start_pixel = 0

    return reconstructed_audio_normalized

def main():
    contrast_scale = 0.15
    MaxMag = get_max_magnitude()

    # Load the audio file
    audio_path = 'GI_GMF_B3_353_20140520_n.wav'
    y, sr = librosa.load(audio_path, sr=None)

    # Plot the audio signal
    plot_audio_signal(y, sr, 'Original Audio Signal')

    # Compute the Fourier Transform
    fft, frequency, magnitude, phase = get_fourier_transform(y, sr)

    # Save magnitude and phase as an image
    audio_to_image(magnitude, phase, contrast_scale, MaxMag)

    # Convert the image back to audio with modified pixels
    factor = 1.7  # change this value as needed
    reconstructed_audio = image_to_audio(sr, contrast_scale, MaxMag, factor)

    # Plot the reconstructed audio signal
    plot_audio_signal(reconstructed_audio, sr, 'Reconstructed Audio Signal')

if __name__ == "__main__":
    main()