import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile

# Load the audio file
audio_file = 'output_audio.wav'
y, sr = librosa.load(audio_file)

# Plot the waveform
plt.figure(figsize=(5, 5))  # Set figsize to create a square image
plt.plot(np.linspace(0, len(y) / sr, num=len(y)), y)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform of Audio")
plt.tight_layout()
plt.show()

# Compute the Fourier Transform
fft = np.fft.fft(y)
magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

plt.figure(figsize=(5, 5))
plt.plot(frequency[:len(frequency)//2], magnitude[:len(magnitude)//2])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Fourier Transform of Audio")
plt.show()

# Show the audio waveform of the Fourier transformed sound clip
y_ifft = np.fft.ifft(fft)  # Inverse Fourier Transform
plt.figure(figsize=(5, 5))
plt.plot(np.linspace(0, len(y_ifft) / sr, num=len(y_ifft)), np.real(y_ifft))  # Take real part after inverse transform
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform of Inverse Fourier Transformed Audio")
plt.tight_layout()
plt.show()

# Save the audio as a new file
output_file = 'transformed_audio.wav'
wavfile.write(output_file, sr, np.real(y_ifft).astype(np.float32))