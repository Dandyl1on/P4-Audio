
# Exploring how magnitude and phase affect a signal

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def load_audio(file_path):
    return wavfile.read(file_path)

def normalize_signal(signal):
    return signal / np.max(np.abs(signal))

def plot_signal(signal, sample_rate, title):
    time = np.arange(len(signal)) / sample_rate
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

def apply_phase_and_magnitude_modifications(signal, phase_shift_deg, magnitude_factor):
    phase_shift_rad = np.deg2rad(phase_shift_deg)
    n = len(signal)
    t = np.arange(n)
    phase = 2 * np.pi * t / n
    phase_shifted_signal = signal * np.exp(1j * phase_shift_rad * phase)
    magnitude = np.abs(phase_shifted_signal)
    phase = np.angle(phase_shifted_signal)
    modified_magnitude = magnitude * magnitude_factor
    modified_signal = modified_magnitude * np.exp(1j * phase)
    return modified_signal

def main():
    file_path = "GI_GMF_B3_353_20140520_n.wav"

    # Load audio file
    sample_rate, signal = load_audio(file_path)

    # Normalize signal
    normalized_signal = normalize_signal(signal)

    # Apply phase shift and modify magnitude
    phase_shift_deg = 0
    magnitude_factor = 1
    modified_signal = apply_phase_and_magnitude_modifications(normalized_signal, phase_shift_deg, magnitude_factor)

    # Plot original and modified signals
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plot_signal(normalized_signal, sample_rate, 'Original Signal')
    plt.subplot(2, 1, 2)
    plot_signal(np.real(modified_signal), sample_rate, f'Modified Signal (Phase Shift: {phase_shift_deg} degrees, Magnitude Factor: {magnitude_factor})')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
