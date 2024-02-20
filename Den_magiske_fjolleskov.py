import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft

def plot_fourier_transform(wav_file, output_png):
    # Read the WAV file
    sample_rate, data = wavfile.read(wav_file)

    # Calculate the number of samples for 1.486s at the given sample rate
    num_samples = int(1.486 * sample_rate)

    # Trim or zero-pad the data to ensure it's the required length
    if len(data) < num_samples:
        # Zero-pad if data is shorter than required
        data_padded = np.pad(data, (0, num_samples - len(data)), mode='constant')
    else:
        # Trim if data is longer than required
        data_padded = data[:num_samples]

    # Calculate the Fourier Transform
    fft_data = fft(data_padded)

    # Frequency bins
    freq = np.fft.fftfreq(len(fft_data), 1 / sample_rate)

    # Create a square image with dimensions 256x256 pixels
    image_size = 256
    image = np.zeros((image_size, image_size, 3))  # 3 channels for magnitude, phase, and combined

    # Assign Fourier transform data to the image
    for i in range(image_size):
        # Magnitude
        image[:, i, 0] = np.abs(fft_data[i * (len(fft_data) // image_size)])
        # Phase (normalized to [0, 1] for better visualization)
        phase = np.angle(fft_data[i * (len(fft_data) // image_size)])
        phase_normalized = (phase + np.pi) / (2 * np.pi)
        image[:, i, 1] = phase_normalized

        # Combined magnitude and phase (for visualization)
        image[:, i, 2] = image[:, i, 0] * (1 - image[:, i, 1])  # Magnitude controls brightness, phase controls hue

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.imshow(image, aspect='auto')
    plt.title('Fourier Transform of ' + wav_file)
    plt.axis('off')

    # Save the plot as PNG
    plt.savefig(output_png, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Fourier transformation saved as {output_png}")


if __name__ == "__main__":
    input_wav = 'GI_GMF_B3_353_20140520_n.wav'
    output_png = 'output_png.png'
    plot_fourier_transform(input_wav, output_png)
