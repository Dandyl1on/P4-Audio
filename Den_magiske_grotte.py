import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# Step 1: Read the .vaw file and extract the audio data
def read_audio_file(file_path):
    rate, data = wav.read(file_path)
    return rate, data

# Step 2: Perform FFT on the audio data
def perform_fft(audio_data):
    return np.fft.fft(audio_data)

# Step 3: Visualize the FFT result as an image
def plot_fft_result(fft_result):
    plt.figure(figsize=(10, 4))
    plt.plot(np.abs(fft_result))
    plt.title('FFT Result')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.show()

# Step 4: Convert the image back to audio data
def inverse_fft(fft_result):
    return np.fft.ifft(fft_result)

# Step 5: Write the audio data to a new .vaw file
def write_audio_file(file_path, rate, data):
    wav.write(file_path, rate, data.astype(np.int16))

# Main function
def main():
    input_file = 'GI_GMF_B3_353_20140520_n.wav'
    output_file = 'output_audio.wav'

    # Step 1: Read the .vaw file and extract the audio data
    rate, audio_data = read_audio_file(input_file)

    # Step 2: Perform FFT on the audio data
    fft_result = perform_fft(audio_data)

    # Step 3: Visualize the FFT result as an image
    plot_fft_result(fft_result)

    # Step 4: Convert the image back to audio data
    inverse_result = inverse_fft(fft_result)

    # Step 5: Write the audio data to a new .vaw file
    write_audio_file(output_file, rate, inverse_result.real)

if __name__ == "__main__":
    main()
