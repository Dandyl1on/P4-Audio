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

# Save the audio
output_file = 'reconstructed_audio.wav'
wavfile.write(output_file, sr, reconstructed_signal.astype(np.float32))

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

file1_path = audio_file
file2_path = output_file

def compare_audio(file1_path, file2_path):

    is_same = open(“file1_path”, "rb").read() == open(“file2_path”, "rb").read()
    if is_same:
        print('Same')
    else:
        print('Different')



print(message)