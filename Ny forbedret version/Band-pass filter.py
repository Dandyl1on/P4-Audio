import numpy as np
import librosa
from PIL import Image
import soundfile as sf
from main import magnitude_db_max, magnitude_db_min, sr, D

def create_bandpass_mask(image_shape, sr, low_freq, high_freq):
    """Create a band-pass mask for the given frequency range."""
    rows, cols = image_shape
    freqs = np.linspace(0, sr / 2, rows)
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Identify indices within the frequency range
    for i, freq in enumerate(freqs):
        if low_freq <= freq <= high_freq:
            mask[i, :] = 1

    return mask

# Load the combined image and split into magnitude and phase parts
combined_img = Image.open('combined_image.png')
combined_img_np = np.array(combined_img)
height, width = combined_img_np.shape
half_height = height // 2
magnitude_img_np = combined_img_np[:half_height, :]
phase_img_np = combined_img_np[half_height:, :]

# Create the band-pass filter mask
low_freq = 300  # Low cutoff frequency in Hz
high_freq = 3000  # High cutoff frequency in Hz
mask = create_bandpass_mask(magnitude_img_np.shape, sr, low_freq, high_freq)

# Apply the mask to the magnitude image
filtered_magnitude_img_np = magnitude_img_np * mask

# Combine the filtered magnitude image with the unchanged phase image
new_combined_img = Image.new('L', (width, height))
new_combined_img.paste(Image.fromarray(filtered_magnitude_img_np), (0, 0))
new_combined_img.paste(Image.fromarray(phase_img_np), (0, half_height))
new_combined_img.save('filtered_combined_image.png')

# Load the new combined image and extract magnitude and phase
combined_img_np = np.array(Image.open('filtered_combined_image.png'))
magnitude_img_np = combined_img_np[:half_height, :]
phase_img_np = combined_img_np[half_height:, :]
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
sf.write('reconstructed_audio_from_filtered_image.wav', y_reconstructed, sr)

print("Reconstructed audio with band-pass filter saved.")
