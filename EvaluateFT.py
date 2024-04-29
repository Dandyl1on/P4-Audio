
# Represent and evaluate the Fourier Transform of an audio signal

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def load_audio(file_path):
    return wavfile.read(file_path)


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
    magnitude = magnitude[:len(magnitude) // 2]

    frequency = (sr / 2) * (np.arange(len(magnitude)) + 1) / len(magnitude)

    return fft, frequency, magnitude

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


def main():
    audio_path = 'GI_GMF_B3_353_20140520_n.wav'
    sr, y = load_audio(audio_path)

    plot_audio_signal(y, sr, 'Original Audio Signal')

    fft, frequency, magnitude = get_fourier_transform(y, sr)

    plot_fourier_transform(frequency, magnitude)

    inverse_ft = inverse_fourier_transform(fft)

    plot_audio_signal(inverse_ft, sr, 'Inverse Fourier Transform')

    # wavfile.write('Inverse_FT.wav', sr, inverse_ft)

if __name__ == "__main__":
    main()
