
# Right image, right sound (?)



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
    magnitude = magnitude[:len(magnitude)//2]

    phase = np.angle(fft)
    phase = phase[:len(phase) // 2]

    frequency = (sr / 2) * (np.arange(len(magnitude)) + 1) / len(magnitude)

    return fft, frequency, magnitude, phase

def get_max_magnitude():
    sine_wave = np.tile([-1, 1], 32768)
    sine_wave = np.concatenate((sine_wave, [1]))

    fft_result = np.fft.fft(sine_wave)
    fft_result = fft_result[1:]

    real_part = np.real(fft_result)
    imag_part = np.imag(fft_result)

    real_grid = np.zeros((128, 256))
    imag_grid = np.zeros((128, 256))

    for k in range(128):
        real_grid[k, :] = real_part[256 * k:256 * k + 256]
        imag_grid[k, :] = imag_part[256 * k:256 * k + 256]

    combined_grid = np.concatenate((real_grid, imag_grid), axis=0)

    max_magnitude = np.max(np.abs(combined_grid))

    return max_magnitude

def audio_to_image(magnitude, phase, contrast_scale, MaxMag):

    magnitude = np.log(1 + magnitude)
    magnitude = magnitude / np.log(MaxMag)

    magnitude = magnitude ** contrast_scale

    phase = phase / np.pi
    phase = phase + 1
    phase = phase / 2

    magnitude = magnitude * 65536
    phase = phase * 65536

    magnitude_image = magnitude.reshape(128, 256)
    phase_image = phase.reshape(128,256)

    combined_image = np.vstack((magnitude_image, phase_image))

    combined_image = Image.fromarray(combined_image.astype(np.uint16))

    combined_image.save('Output_Image.png')

    return combined_image

def image_to_audio(contrast_scale, MaxMag):

    image = Image.open('Output_Image.png')
    image_array = np.array(image)
    image_array = np.double(image_array)

    magnitude = image_array[:128, :]
    magnitude = magnitude.reshape(-1)

    magnitude = magnitude / 65536
    magnitude = magnitude ** (1 / contrast_scale)
    magnitude = magnitude * np.log(MaxMag)
    magnitude = np.exp(magnitude) - 1

    phase = image_array[128:, :]
    phase = phase.reshape(-1)
    phase = phase / 65536
    phase = phase * 2
    phase = phase - 1
    phase = phase * np.pi

    reversed_magnitude = np.flip(magnitude)
    magnitude = np.concatenate((magnitude, reversed_magnitude))
    reversed_phase = np.flip(phase)
    phase = np.concatenate((phase, -reversed_phase))

    fft = magnitude * np.exp(1j * phase)

    reconstructed_audio = np.fft.ifft(fft).real

    reconstructed_audio_normalized = reconstructed_audio / np.max(np.abs(reconstructed_audio))

    return reconstructed_audio_normalized

def main():

    bit16 = 65536

    contrast_scale = 0.25

    MaxMag = get_max_magnitude()

    audio_path = 'GI_GMF_B3_353_20140520_n.wav'
    y, sr = librosa.load(audio_path, sr=None)

    plot_audio_signal(y, sr, 'Original Audio Signal')

    fft, frequency, magnitude, phase = get_fourier_transform(y, sr)

    audio_to_image(magnitude, phase, contrast_scale, MaxMag)

    reconstructed_audio = image_to_audio(contrast_scale, MaxMag)

    plot_audio_signal(reconstructed_audio, sr, 'Reconstructed Audio Signal')

if __name__ == "__main__":
    main()