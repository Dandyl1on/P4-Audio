import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt


def plot_polar_spectrums(freqs, magnitudes, phases):
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(211, projection='polar')
    ax1.scatter(phases, magnitudes, color='b', marker='.')
    ax1.set_title('Polar Magnitude Spectrum')

    ax2 = fig.add_subplot(212, projection='polar')
    ax2.scatter(phases, np.ones_like(phases), color='r', marker='.')
    ax2.set_title('Polar Phase Spectrum')

    plt.tight_layout()
    plt.show()


def main(audio_file):
    # Read audio file
    sample_rate, data = wavfile.read(audio_file)

    # Compute the Fourier transform
    frequencies, amplitudes = signal.welch(data, fs=sample_rate, nperseg=1024)

    # Convert to polar coordinates
    magnitudes = np.abs(amplitudes)
    phases = np.angle(amplitudes)

    # Plot polar spectrums
    plot_polar_spectrums(frequencies, magnitudes, phases)

if __name__ == "__main__":
    audio_file = "GI_GMF_B3_353_20140520_n.wav"
    main(audio_file)
