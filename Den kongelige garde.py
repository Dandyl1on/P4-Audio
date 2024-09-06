import numpy as np
import librosa
from PIL import Image
import soundfile as sf

def reshape_to_custom(array, column_length):
    """ Reshape the array to have a fixed number of columns and calculate rows accordingly. """
    num_elements = array.size
    num_cols = column_length
    num_rows = int(np.ceil(num_elements / num_cols))
    # Create a new array with padding to fit the new size
    resized_array = np.zeros((num_rows, num_cols))
    resized_array.flat[:num_elements] = array.flat
    return resized_array

# Load audio file and compute STFT
audio_path = 'GI_GMF_B3_353_20140520_n.wav'
y, sr = librosa.load(audio_path, sr=None)
D = librosa.stft(y)
magnitude = np.abs(D)
phase = np.angle(D)

# Calculate the column length as twice the square root of the array length
column_length = int(1.5 * np.ceil(np.sqrt(magnitude.size)))

# Reshape magnitude and phase arrays
magnitude_resized = reshape_to_custom(magnitude, column_length)
phase_resized = reshape_to_custom(phase, column_length)

# Normalize magnitude to a logarithmic scale and then to [0, 1]
magnitude_db = 20 * np.log10(np.maximum(magnitude_resized, 1e-10))
magnitude_db_min = np.min(magnitude_db)
magnitude_db_max = np.max(magnitude_db)
magnitude_normalized = (magnitude_db - magnitude_db_min) / (magnitude_db_max - magnitude_db_min)

# Normalize phase to [0, 1]
phase_normalized = (phase_resized + np.pi) / (2 * np.pi)

# Convert arrays to images
magnitude_image = Image.fromarray((magnitude_normalized * 255).astype(np.uint8), mode='L')
phase_image = Image.fromarray((phase_normalized * 255).astype(np.uint8), mode='L')

# Save images
magnitude_image.save('magnitude_image.png')
phase_image.save('phase_image.png')

# Load and stack images vertically
magnitude_img = Image.open('magnitude_image.png')
phase_img = Image.open('phase_image.png')

# Create a new image with the combined height of both images
combined_img = Image.new('L', (magnitude_img.width, magnitude_img.height + phase_img.height))
combined_img.paste(magnitude_img, (0, 0))
combined_img.paste(phase_img, (0, magnitude_img.height))
combined_img.save('combined_image.png')

# Load combined image and extract magnitude and phase
combined_img_np = np.array(Image.open('combined_image.png'))
magnitude_img_np = combined_img_np[:magnitude_img.height, :]
phase_img_np = combined_img_np[magnitude_img.height:, :]
magnitude_normalized = magnitude_img_np / 255.0
phase_normalized = phase_img_np / 255.0

# Denormalize magnitude
magnitude_db = magnitude_normalized * (magnitude_db_max - magnitude_db_min) + magnitude_db_min
magnitude = 10**(magnitude_db / 20)  # Convert dB back to amplitude

# Denormalize phase
phase = phase_normalized * (2 * np.pi) - np.pi

# Flatten the arrays back to their original shape
magnitude_flat = magnitude.flatten()[:D.size]
phase_flat = phase.flatten()[:D.size]

# Reshape to original STFT shape
magnitude_reshaped = magnitude_flat.reshape(D.shape)
phase_reshaped = phase_flat.reshape(D.shape)

# Reconstruct complex STFT and audio signal
D_reconstructed = magnitude_reshaped * np.exp(1j * phase_reshaped)
y_reconstructed = librosa.istft(D_reconstructed)

# Save the reconstructed audio
sf.write('reconstructed_audio_from_combined_image.wav', y_reconstructed, sr)

print("Reconstructed audio saved.")
