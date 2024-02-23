import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

def Change_Amplitude(fourier_transform):
    start_index = 1
    end_index = 1000
    modification_amount = 10  # Change the amplitude by this factor

    for index in range(start_index, end_index + 1):  # Adding 1 to end_index to make it inclusive
        fourier_transform[index] =+ modification_amount

def Change_Phase(fourier_transform):
    # Modify one value in the amplitude and phase slightly
    PhaseS_index = 1
    PhaseE_index = 1000
    modification_amount_phase = np.pi / 180  # Change the phase by this angle (45 degrees)

    # Modify amplitude
    for index in range(PhaseS_index, PhaseE_index + 1):
        phase_shift = np.angle(fourier_transform[index]) + modification_amount_phase
        fourier_transform[index] = np.abs(fourier_transform[index]) * np.exp(1j * phase_shift)

def plot_spectrum(wav_file, output_wav_file):
    # Read the WAV file
    sample_rate, data = wavfile.read(wav_file)

    # Compute the Fourier Transform
    fourier_transform = np.fft.fft(data)

    Change_Amplitude(fourier_transform)

    Change_Phase(fourier_transform)

    # Compute the frequencies corresponding to the Fourier Transform
    frequencies = np.fft.fftfreq(len(data)) * sample_rate

    # Plot the spectrum in Cartesian coordinates
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.plot(frequencies, np.abs(fourier_transform))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency Spectrum')
    plt.grid(True)

    # Plot the spectrum in polar coordinates
    plt.subplot(1, 3, 2, polar=True)
    plt.plot(np.angle(fourier_transform), np.abs(fourier_transform))
    plt.title('Polar Spectrum')
    plt.grid(True)

    # Plot the phase spectrum
    plt.subplot(1, 3, 3)
    plt.plot(frequencies, np.angle(fourier_transform))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.title('Phase Spectrum')
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