from __future__ import print_function
import math
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
from scipy.io import wavfile # get the api
import scipy

sound = AudioSegment.from_wav("GI_GMF_B3_353_20140520_n.wav")
play(sound)

# Extract raw audio data as numpy array
raw_data = sound.get_array_of_samples()

# Perform Fourier transform
transformed_data = np.fft.fft(raw_data)

# Play the transformed sound (inverse Fourier transform)
# Inverse transform to get back to the time domain
transformed_sound = AudioSegment(data=np.fft.ifft(transformed_data).real.astype(np.int16), sample_width=2, frame_rate=sound.frame_rate, channels=1)

# Play the transformed sound
play(transformed_sound)
# If you want to visualize the Fourier transform
plt.plot(np.abs(transformed_data))
plt.title('Fourier Transform')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.show()

