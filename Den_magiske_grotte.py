import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

# Step 1: Read the .wav file and extract the audio data
def read_audio_file(file_path):
    rate, data = wav.read(file_path)
    print("Data Shape:", data.shape)
    return rate, data

# Step 2: Perform Fast Fourier Transform (FFT) on the audio data
def perform_fft(audio_data):
    return np.fft.fft(audio_data)

# Step 3: Perform Discrete Cosine Transform (DCT) on the audio data
def perform_dct(audio_data):
    transposed_data = audio_data.T
    dct_result = dct(transposed_data, type=2, axis=0)
    print("DCT Result Shape:", dct_result.shape)
    return dct_result
def plot_fft_spectrogram(fft_result, rate):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.specgram(fft_result, Fs=rate, cmap='viridis')
    plt.title('Magnitude Spectrogram (FFT)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.subplot(2, 2, 2)
    plt.specgram(np.angle(fft_result), Fs=rate, cmap='hsv')
    plt.title('Phase Spectrogram (FFT)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')



# Step 6: Convert the FFT result back to audio data
def inverse_fft(fft_result):
    return np.fft.ifft(fft_result)

# Step 7: Convert the DCT result back to audio data
def inverse_dct(dct_result):
    return idct(dct_result, type=2)

# Step 8: Write the audio data to a new .wav file
def write_audio_file(file_path, rate, data):
    wav.write(file_path, rate, data.astype(np.int16))

# Main function
def main():
    input_file = 'GI_GMF_B3_353_20140520_n.wav'
    output_file_fft = 'output_audio_fft.wav'
    output_file_dct = 'output_audio_dct.wav'

    # Step 1: Read the .wav file and extract the audio data
    rate, audio_data = read_audio_file(input_file)

    # Step 2: Perform Fast Fourier Transform (FFT) on the audio data
    fft_result = perform_fft(audio_data)

    # Step 3: Perform Discrete Cosine Transform (DCT) on the audio data
    dct_result = perform_dct(audio_data)

    # Step 4: Visualize the FFT result as a spectrogram
    plot_fft_spectrogram(fft_result, rate)

    # Step 6: Convert the FFT result back to audio data and write to a new .wav file
    inverse_result_fft = inverse_fft(fft_result)
    write_audio_file(output_file_fft, rate, inverse_result_fft.real)

    # Step 7: Convert the DCT result back to audio data and write to a new .wav file
    inverse_result_dct = inverse_dct(dct_result)
    write_audio_file(output_file_dct, rate, inverse_result_dct)

if __name__ == "__main__":
    main()
