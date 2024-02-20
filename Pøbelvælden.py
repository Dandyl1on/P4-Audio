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

# Plot the original audio signal
plt.figure(figsize=(10, 4))
plt.plot(np.arange(len(y)) / sr, y)
plt.title('Original Audio Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# Plot the Fourier Transform
plt.figure(figsize=(10, 4))
plt.plot(frequency, magnitude)
plt.title('Fourier Transform')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()

# Plot the Reconstructed Audio Signal
plt.figure(figsize=(10, 4))
plt.plot(np.arange(len(reconstructed_signal)) / sr, reconstructed_signal)
plt.title('Reconstructed Audio Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# Evaluation: The Fourier Transform of the input audio signal has observable "folding" or "aliasing" in the frequency domain. This is caused by the presence of frequency components that exist beyond the Nyquist frequency. The Nyquist frequency is the highest frequency that can be accurately represented by a digital signal, the value of which is half of the sampling rate of the signal. Frequencies that exist beyond the Nyquist frequency may cause aliasing. Aliasing occurs when frequencies above the Nyquist frequency are folded back into the audible frequency range. The solution to this aliasing issue would be to bandlimit the input signal or apply anti-aliasing filters, however, as can be seen in the time-domain representation of the reconstructed audio, the aliasing does not seem to have a visually observable effect.

