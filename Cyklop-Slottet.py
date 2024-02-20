import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile
import pygame

blue_image = pygame.image.load('image.png')

# Downsample the image
downsample_factor = 4
blue_image = pygame.transform.scale(blue_image, (blue_image.get_width() // downsample_factor,
                                                 blue_image.get_height() // downsample_factor))

# Extract dimensions
width, height = blue_image.get_size()

# Initialize Pygame mixer
pygame.mixer.init()

# Set sound parameters
sample_rate = 44100
duration = 1.5  # in seconds
volume = 0.5  # 0.0 to 1.0

# Initialize sound array
sound_data = np.zeros((int(sample_rate * duration), 2), dtype=np.float32)

# Convert image to sound
blue_intensity = np.mean(pygame.surfarray.array3d(blue_image)[:, :, 2])
frequency = 6 + (blue_intensity * 10000)  # Adjust this range as needed
phase = 0
for t in range(int(sample_rate * duration)):
    sound_data[t, 0] = volume * np.sin(2 * np.pi * frequency * t / sample_rate + phase)
    sound_data[t, 1] = volume * np.sin(2 * np.pi * frequency * t / sample_rate + phase)

# Normalize sound data
max_value = np.max(np.abs(sound_data))
sound_data /= max_value

# Convert to bytes
sound_bytes = (sound_data * 32767).astype(np.int16).tobytes()

# Play the sound
print("Playing Sound")
pygame.mixer.Sound(sound_bytes).play()

# Wait for the sound to finish
pygame.time.wait(int(duration * 1000))
print("Finished Playing Sound")
