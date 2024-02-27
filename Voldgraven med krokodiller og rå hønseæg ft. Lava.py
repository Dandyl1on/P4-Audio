import numpy as np
from PIL import Image
import scipy.fftpack as fft
import soundfile as sf
import pygame

def polar_to_rect(mag, phase):
    # Convert polar coordinates to rectangular coordinates
    return mag * np.exp(1j * phase)

def image_to_audio(image_path, audio_path, sample_rate=44100):
    # Load the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = np.array(img)

    # Split polar coordinates
    mag = img[:, :img.shape[1]//2]
    phase = img[:, img.shape[1]//2:]

    # Convert polar coordinates to rectangular coordinates
    f_transform = polar_to_rect(mag, phase)

    # Inverse Fourier Transform
    inv_transform = np.abs(fft.ifft2(f_transform))

    # Normalize values to fit into audio range
    inv_transform = inv_transform / np.max(inv_transform)
    inv_transform = (inv_transform * 32767).astype(np.int16)

    # Save as audio file
    sf.write(audio_path, inv_transform, sample_rate)

def play_audio(audio_path):
    # Load the audio data
    data, sample_rate = sf.read(audio_path, dtype='int16')

    # Initialize pygame mixer
    pygame.mixer.init(sample_rate)

    # Convert data to raw bytes
    raw = data.tobytes()

    # Load the sound
    sound = pygame.mixer.Sound(buffer=raw)

    # Play the sound
    sound.play()

    # Wait until sound is finished playing
    pygame.time.wait(int(sound.get_length() * 1000))

if __name__ == "__main__":
    image_path = "spectrogram.png"
    audio_path = "HubbaBubbaBirthday3.aiff"
    image_to_audio(image_path, audio_path)
    print("Audio file saved successfully.")
    play_audio(audio_path)
