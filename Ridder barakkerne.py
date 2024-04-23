import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile
from PIL import Image
import math

def GetMaxMagnitude():
    A = np.tile([-1, 1], 32768)
    A = np.concatenate((A, [1]))

    B = np.fft.fft(A)
    B = B[1:]

    B = B[:len(B) // 2]

    R = np.real(B)
    I = np.imag(B)

    RR = np.zeros((128, 256))
    II = np.zeros((128, 256))

    for k in range(128):
        RR[k, :] = R[256 * k:256 * k + 256]
        II[k, :] = I[256 * k:256 * k + 256]

    AA = np.concatenate((RR, II), axis=0)
    F = np.max(np.abs(AA))

    return F

def plot_audio_signal(y, sr, name):
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(y)) / sr, y)
    plt.title(name)
    plt.xlabel('Time (s)')
    plt.ylabel('Magnitude')
    plt.tight_layout()
    plt.show()

def get_fourier_transform(y, sr):
    fft = np.fft.fft(y)

    magnitude = np.abs(fft)
    magnitude = magnitude[:len(magnitude)//2]
    phase = np.angle(fft)
    phase = phase[:len(phase) // 2]
    frequency = (sr / 2) * (np.arange(len(magnitude)) + 1) / len(magnitude)
    return fft, frequency, magnitude, phase

def plot_fourier_transform(frequency, magnitude):
    plt.figure(figsize=(12, 8))
    plt.plot(frequency, magnitude)
    plt.title('Fourier Transform')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.tight_layout()
    plt.show()

def inverse_fourier_transform(fft_signal):
    inverse_FT_transform = np.fft.ifft(fft_signal)
    return inverse_FT_transform.real

def plot_phase(frequency, phase):
    plt.figure(figsize=(10, 4))
    plt.plot(frequency, phase)
    plt.title('Phase Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase')
    plt.tight_layout()
    plt.show()

# 4    Wrong image, right sound and range

def audio_to_image(magnitude, phase, contrast_scale, MaxMag):

    # first, normalization concerns
    # phase is between -pi to pi and it doesn't change

    # magnitude is a more difficult issue
    # (1) you don't know in advance the max value of magnitude for one file
    # if we normalize the magnitude independently file by file, we have a problem later with NN training
    # indeed, if we do so, the energy values lose their intrinsic meaning
    # therefore, we have to find the max value the magnitude may ever take
    # this max value occurs for the FT of one sine wave
    # it is 20861 (you can evaluate the FT of one sine wave over the same length to confirm)
    # (2) we have to convert magnitude to log scale, otherwise we don't see anything on the image, as most values are near to zero
    # then log(zero) will not work as it is minus infinity
    # so we have to convert to log(1+magnitude) and then normalize

    # plt.plot(magnitude)
    # plt.show()

    # magnitude normalization (done in log scale)
    magnitude = np.log(1 + magnitude)
    magnitude = magnitude / np.log(MaxMag)

    # now, to be able to even better see, we do a non-linear transform on magnitude using the square root transformation operation
    # 0 will remain at 0, 1 will remain at 1, but medium values will be higher
    # Following the logarithmic transformation, the square root operation is applied to increase the contrast of the image, making small components more visibl
    magnitude = magnitude ** contrast_scale

    # plt.plot(magnitude)
    # plt.show()
    # plt.plot(phase)
    # plt.show()

    # phase normalization
    phase = phase / np.pi   # [0, 2], or 0 and 2pi radians
    phase = phase + 1   # [1, 3]
    phase = phase / 2   # [0.5, 1.5] to match magnitude range

    # now, to between 0 and 255 for image representation as pixels
    magnitude = magnitude * 65536
    phase = phase * 65536

    # Reshape magnitude and phase arrays
    magnitude_image = magnitude.reshape(128, 256)
    phase_image = phase.reshape(128,256)

    # Combine magnitude and phase images
    combined_image = np.vstack((magnitude_image, phase_image))
    # print(combined_image.shape)

    # Convert to PIL Image
    combined_image = Image.fromarray(combined_image.astype(np.uint16))   # All details in image may already be covered in 8-bit, thus no change in 16-bit

    # Save the image (optional)
    combined_image.save('Output_Image.png')

    # Print debugging information
    # print("Magnitude (dB) min:", np.min(magnitude))
    # print("Magnitude (dB) max:", np.max(magnitude))

    return combined_image

def image_to_audio(sr, contrast_scale, MaxMag):

    image = Image.open('1st pixel change gray.png')
    image_array = np.array(image)
    image_array = np.double(image_array)

    # extract magnitude
    magnitude = image_array[:128, :]
    magnitude = magnitude.reshape(-1)
    # and now we do the inverse process as when we were making the image
    magnitude = magnitude / 65536
    magnitude = magnitude ** (1 / contrast_scale)
    magnitude = magnitude * np.log(MaxMag)
    magnitude = np.exp(magnitude) - 1

    # plt.plot(magnitude)
    # plt.show()

    # extract phase
    phase = image_array[128:, :]
    phase = phase.reshape(-1)
    phase = phase / 65536
    phase = phase * 2
    phase = phase - 1
    phase = phase * np.pi

    # reconstruct complete amplitude and phase
    reversed_magnitude = np.flip(magnitude)
    magnitude = np.concatenate((magnitude, reversed_magnitude))
    reversed_phase = np.flip(phase)
    phase = np.concatenate((phase, -reversed_phase))

    # Combine magnitude and phase
    fft = magnitude * np.exp(1j * phase)

    # Inverse Fourier Transform
    reconstructed_audio = np.fft.ifft(fft).real

    # Normalize reconstructed audio
    reconstructed_audio_normalized = reconstructed_audio / np.max(np.abs(reconstructed_audio))

    # Print debugging information
    #print("Reconstructed audio min:", np.min(reconstructed_audio_normalized))
    #print("Reconstructed audio max:", np.max(reconstructed_audio_normalized))

    return reconstructed_audio_normalized

def main():

    contrast_scale = 0.25

    MaxMag = GetMaxMagnitude()

    # Load the audio file
    audio_path = 'GI_GMF_B3_353_20140520_n.wav'
    y, sr = librosa.load(audio_path, sr=None)

    # Plot the audio signal
    plot_audio_signal(y, sr, 'Original Audio Signal')

    # Compute the Fourier Transform
    fft, frequency, magnitude, phase = get_fourier_transform(y, sr)

    # Plot the Fourier Transform
    # plot_fourier_transform(frequency, magnitude)

    # Plot the phase spectrum
    # plot_phase(frequency, phase)

    # Save magnitude and phase as an image
    audio_to_image(magnitude, phase, contrast_scale, MaxMag)

    # Convert the image back to audio
    reconstructed_audio = image_to_audio(sr, contrast_scale, MaxMag)

    # Plot the reconstructed audio signal
    plot_audio_signal(reconstructed_audio, sr, 'Reconstructed Audio Signal')

    # Save the reconstructed audio signal
    wavfile.write('ChangeName.wav', sr, reconstructed_audio)

if __name__ == "__main__":
    main()