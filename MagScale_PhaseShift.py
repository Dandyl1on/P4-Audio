

# Exploring how magnitude and phase affect a signal


import numpy as np
import matplotlib.pyplot as plt
import librosa

def plot_audio_signal(y, sr, name):
    time = np.arange(len(y)) / sr
    plt.plot(time, y)
    plt.title(name)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
def apply_modifications(y, phase_shift_deg, magnitude_factor):

    phase_shift_rad = np.deg2rad(phase_shift_deg)

    nr_samples = len(y)
    time_array = np.arange(nr_samples)
    phase = 2 * np.pi * time_array / nr_samples
    phase_shifted_signal = y * np.exp(1j * phase_shift_rad * phase)

    magnitude = np.abs(phase_shifted_signal)

    phase = np.angle(phase_shifted_signal)

    modified_magnitude = magnitude * magnitude_factor
    modified_signal = modified_magnitude * np.exp(1j * phase)

    return modified_signal
def get_fourier_transform(y, sr):
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    magnitude = magnitude[:len(magnitude)//2]
    phase = np.angle(fft)
    phase = phase[:len(phase) // 2]
    frequency = (sr / 2) * (np.arange(len(magnitude)) + 1) / len(magnitude)
    return fft, frequency, magnitude, phase
def plot_pc(frequency, magnitude, phase):
    plt.figure(figsize=(12, 8))
    plt.polar(phase, magnitude, markersize=1)
    plt.title('Polar Coordinates of Fourier Transform')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(frequency[:len(phase)], phase, markersize=1)
    plt.title('Phase Information')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase')
    plt.tight_layout()
    plt.show()
def main():
    audio_path = 'GI_GMF_B3_353_20140520_n.wav'
    y, sr = librosa.load(audio_path, sr=None)

    phase_shift_deg = 90
    magnitude_factor = 2
    modified_signal = apply_modifications(y, phase_shift_deg, magnitude_factor)

    fft, frequency, magnitude, phase = get_fourier_transform(modified_signal, sr)

    plot_pc(frequency, magnitude, phase)
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plot_audio_signal(y, sr, 'Original Signal')
    plt.subplot(2, 1, 2)
    plot_audio_signal(np.real(modified_signal), sr, f'Modified Signal (Phase Shift: {phase_shift_deg} degrees,'
                                                    f' Magnitude Factor: {magnitude_factor})')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
