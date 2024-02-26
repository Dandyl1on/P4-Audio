import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevent showing plot in interactive window
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

def polar_to_image(filename, image_size):
  """
  Converts the polar coordinates of the Fourier transform of an audio file to a square image.

  Args:
      filename: Path to the audio file.
      image_size: Size of the output image (square).

  Returns:
      A numpy array representing the image data.
  """
  # Read audio data
  sample_rate, audio_data = read(filename)

  # Perform Fourier transform
  fft_data = np.fft.fft(audio_data)

  # Get magnitude and phase (polar coordinates)
  magnitude = np.abs(fft_data)
  phase = np.angle(fft_data)

  # Create square image from half of the magnitude data (excluding negative frequencies)
  image = np.zeros((image_size, image_size))
  half_size = image_size // 2

  # Fill image with magnitude data (skipping DC component at index 0)
  image[:half_size, :half_size] = magnitude[1:half_size + 1]

  # Mirror magnitude for the other half of the image (assuming real audio signal)
  image[half_size:, :half_size] = np.flipud(image[:half_size, :half_size])

  # Mirror magnitude for the remaining half of the image (assuming real audio signal)
  image[:half_size, half_size:] = np.fliplr(image[:half_size, :half_size])

  # Mirror magnitude for the remaining quadrant (assuming real audio signal)
  image[half_size:, half_size:] = np.flipud(np.fliplr(image[:half_size, :half_size]))

  # Normalize image data (optional)
  image = image / np.max(image)

  return image

# Example usage
filename = "GI_GMF_B3_353_20140520_n.wav"
image_size = 256
image_data = polar_to_image(filename, image_size)

# Save the image without showing the plot
plt.imsave("spectrogram.png", image_data, cmap="gray")  # Replace "spectrogram.png" with your desired filename
