

# Evaluate and represent Fourier Transform


import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile

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

    plt.figure(figsize=(12, 16))

    plt.subplot(2, 1, 1)
    plt.plot(frequency, magnitude)
    plt.title('Fourier Transform')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    magnitude_db = 20 * np.log10(magnitude)
    plt.plot(frequency, magnitude_db)
    plt.title('Fourier Transform (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.tight_layout()

    plt.show()

def main():

    audio_path = 'GI_GMF_B3_353_20140520_n.wav'
    y, sr = librosa.load(audio_path, sr=None)

    plot_audio_signal(y, sr, 'Original Audio Signal')

    fft, frequency, magnitude = get_fourier_transform(y, sr)

    plot_fourier_transform(frequency, magnitude)

    inverse_ft = np.fft.ifft(fft).real

    plot_audio_signal(inverse_ft, sr, 'Inverse Fourier Transform')

    wavfile.write('Inverse_FT.wav', sr, inverse_ft)

if __name__ == "__main__":
    main()
