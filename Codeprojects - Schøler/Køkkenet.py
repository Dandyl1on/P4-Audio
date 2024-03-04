import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt


def audio_to_polar_image(audio_file, magnitude_scale=1.0, image_size=512):
    # Read audio file
    sample_rate, data = scipy.io.wavfile.read(audio_file)

    # Compute Fourier Transform
    fourier_transform = np.fft.fft(data)

    # Scale magnitude
    fourier_transform *= magnitude_scale

    # Convert Fourier Transform to polar coordinates
    magnitude = np.abs(fourier_transform)
    phase = np.angle(fourier_transform)

    # Interpolate polar coordinates to make a square image
    r, theta = np.meshgrid(np.linspace(0, 1, image_size), np.linspace(0, 2 * np.pi, image_size), indexing='ij')
    interp_magnitude = np.interp(r.flatten(), np.linspace(0, 1, magnitude.shape[0]), magnitude)
    interp_magnitude = interp_magnitude.reshape(image_size, image_size)
    interp_phase = np.interp(r.flatten(), np.linspace(0, 1, phase.shape[0]), phase)
    interp_phase = interp_phase.reshape(image_size, image_size)

    # Convert polar coordinates to rectangular coordinates
    X = interp_magnitude * np.cos(interp_phase)
    Y = interp_magnitude * np.sin(interp_phase)

    # Plot the image
    plt.figure(figsize=(8, 8))
    plt.imshow(np.abs(X + 1j * Y), cmap='gray', aspect='auto')
    plt.axis('off')
    plt.show()


audio_file = 'GI_GMF_B3_353_20140520_n.wav'
audio_to_polar_image(audio_file, magnitude_scale=10.0)
