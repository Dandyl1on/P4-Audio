# You are all peasents

import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile

# Load the audio file
audio_file = 'GI_GMF_B3_353_20140520_n.wav'
y, sr = librosa.load(audio_file)

# Compute the Fourier Transform
fft = np.fft.fft(y)
magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

# Inverse Fourier Transform
reconstructed_signal = np.fft.ifft(fft).real

# Plot the original audio, Fourier Transform, and reconstructed audio signals
plt.figure(figsize=(14, 10))

# Original Audio
plt.subplot(3, 1, 1)
plt.plot(np.arange(len(y)) / sr, y)
plt.title('Original Audio Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Fourier Transform
plt.subplot(3, 1, 2)
plt.plot(frequency, magnitude)
plt.title('Fourier Transform')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

# Reconstructed Audio
plt.subplot(3, 1, 3)
plt.plot(np.arange(len(reconstructed_signal)) / sr, reconstructed_signal)
plt.title('Reconstructed Audio Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()