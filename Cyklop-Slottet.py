import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def plot_spectrum(wav_file, output_wav_file):
    # Read the WAV file
    sample_rate, data = wavfile.read(wav_file)

    # Compute the Fourier Transform
    fourier_transform = np.fft.fft(data)

    # Modify one value in the amplitude slightly
    index_to_modify = 1000  # Modify the value at this index
    modification_amount = 1000  # Change the amplitude by this factor
    fourier_transform[index_to_modify] *= modification_amount

    # Compute the frequencies corresponding to the Fourier Transform
    frequencies = np.fft.fftfreq(len(data)) * sample_rate

    # Plot the spectrum in Cartesian coordinates
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(frequencies, np.abs(fourier_transform))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency Spectrum')
    plt.grid(True)

    # Plot the spectrum in polar coordinates
    plt.subplot(1, 2, 2, polar=True)
    plt.plot(np.angle(fourier_transform), np.abs(fourier_transform))
    plt.title('Polar Spectrum')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save transformed audio
    transformed_audio = np.fft.ifft(fourier_transform).real.astype(np.int16)
    wavfile.write(output_wav_file, sample_rate, transformed_audio)
    print(f"Transformed audio saved as {output_wav_file}")

if __name__ == "__main__":
    # Provide the path to your WAV file
    wav_file = "GI_GMF_B3_353_20140520_n.wav"
    output_wav_file = "FourierTransformed.wav"
    plot_spectrum(wav_file, output_wav_file)