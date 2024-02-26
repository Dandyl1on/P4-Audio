import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def compute_inverse_fourier_transform(wav_file, output_wav_file):
    # Read the WAV file
    sample_rate, data = wavfile.read(wav_file)

    # Compute the Fourier Transform
    fourier_transform = np.fft.fft(data)

    # Compute the inverse Fourier Transform
    inverse_fourier_transform = np.fft.ifft(fourier_transform)

    # Convert to real values and cast to 16-bit integer (assuming audio is in int16 format)
    inverse_audio = inverse_fourier_transform.real.astype(np.int16)

    # Save the inverse Fourier transformed audio
    wavfile.write(output_wav_file, sample_rate, inverse_audio)
    print(f"Inverse Fourier transformed audio saved as {output_wav_file}")

if __name__ == "__main__":
    # Provide the path to your WAV file and specify the output file name
    wav_file = r"C:\Users\nicol\Desktop\Code Projects\Python\P4-Audio\P4-Audio\Codeprojects - Schøler\FourierTransformed.wav"
    output_wav_file = "InverseFourierTransformed.wav"
    compute_inverse_fourier_transform(wav_file, output_wav_file)
