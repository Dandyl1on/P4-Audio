import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import spectrogram

def plot_audio_waveform(audio_data, sample_rate):
    time = np.arange(0, len(audio_data)) / sample_rate
    plt.figure(figsize=(10, 4))
    plt.plot(time, audio_data)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform')
    plt.show()


def plot_magnitude_spectrum(audio_data, sample_rate):
    fft_output = np.fft.fft(audio_data)
    magnitude_spectrum = np.abs(fft_output)
    frequency = np.fft.fftfreq(len(audio_data), 1 / sample_rate)
    plt.figure(figsize=(10, 4))
    plt.plot(frequency[:len(frequency) // 2], magnitude_spectrum[:len(magnitude_spectrum) // 2])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Magnitude Spectrum')
    plt.show()


def plot_phase_spectrum(audio_data, sample_rate):
    fft_output = np.fft.fft(audio_data)
    phase_spectrum = np.angle(fft_output)
    frequency = np.fft.fftfreq(len(audio_data), 1 / sample_rate)
    plt.figure(figsize=(10, 4))
    plt.plot(frequency[:len(frequency) // 2], phase_spectrum[:len(phase_spectrum) // 2])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.title('Phase Spectrum')
    plt.show()


def plot_spectrogram(audio_data, sample_rate):
    f, t, Sxx = spectrogram(audio_data, sample_rate)
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    plt.colorbar(label='Intensity [dB]')
    plt.show()


def main(audio_file):
    sample_rate, audio_data = wav.read(audio_file)

    # Plot the original audio waveform
    plot_audio_waveform(audio_data, sample_rate)

    # Plot the magnitude spectrum
    plot_magnitude_spectrum(audio_data, sample_rate)

    # Plot the phase spectrum
    plot_phase_spectrum(audio_data, sample_rate)

    # Plot the spectogram
    plot_spectrogram(audio_data, sample_rate)


if __name__ == "__main__":
    audio_file = r"C:\Users\nicol\Desktop\Code Projects\Python\P4-Audio\P4-Audio\Codeprojects - Schøler\FourierTransformed.wav"
    main(audio_file)
