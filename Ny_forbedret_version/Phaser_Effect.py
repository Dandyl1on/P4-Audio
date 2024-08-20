
# This version of the all-filters.py with a Phaser Effect applied as of 19/08/2024
# The Phaser Effect is a type of audio effect that introduces phase shifts to the audio signal.
# This effect does not comply with the limitation of only editing the pixels in the image.

import numpy as np
import librosa
from PIL import Image
import soundfile as sf
from main import magnitude_db_max, magnitude_db_min, sr, D

def reshape_to_custom(array, column_length):
    """ Reshape the array to have a fixed number of columns and calculate rows accordingly. """
    num_elements = array.size
    num_cols = column_length
    num_rows = int(np.ceil(num_elements / num_cols))
    resized_array = np.zeros((num_rows, num_cols))
    resized_array.flat[:num_elements] = array.flat
    return resized_array

def create_filter_mask(image_shape, sr, filter_type, cutoff_freqs):
    """Create a filter mask for the given frequency range."""
    rows, cols = image_shape
    freqs = np.linspace(0, sr / 2, rows)
    mask = np.zeros(image_shape, dtype=np.uint8)

    if filter_type == 'bandpass':
        low_freq, high_freq = cutoff_freqs
        mask = np.array([[1 if low_freq <= freq <= high_freq else 0 for _ in range(cols)] for freq in freqs])
    elif filter_type == 'lowpass':
        cutoff_freq = cutoff_freqs[0]
        mask = np.array([[1 if freq <= cutoff_freq else 0 for _ in range(cols)] for freq in freqs])
    elif filter_type == 'highpass':
        cutoff_freq = cutoff_freqs[0]
        mask = np.array([[1 if freq >= cutoff_freq else 0 for _ in range(cols)] for freq in freqs])
    elif filter_type == 'notch':
        notch_freq, bandwidth = cutoff_freqs
        mask = np.array([[0 if (notch_freq - bandwidth/2) <= freq <= (notch_freq + bandwidth/2) else 1 for _ in range(cols)] for freq in freqs])

    return mask

def apply_phaser_effect(phase_img_np, num_notches=6, depth=0.5, sweep_speed=0.1):
    """
    Apply a phaser effect by introducing phase shifts.
    - num_notches: Number of phase shift notches (frequency bands).
    - depth: The strength of the phase shift.
    - sweep_speed: The speed of the sweeping effect across time.
    """
    # Convert phase image to float to allow adding phase shifts
    phase_img_np = phase_img_np.astype(np.float64)

    rows, cols = phase_img_np.shape
    freqs = np.linspace(0, 1, rows)  # Normalized frequency axis
    time = np.linspace(0, 1, cols)  # Normalized time axis

    for i in range(num_notches):
        notch_center = i / num_notches  # Center of each notch
        phase_shift = depth * np.sin(2 * np.pi * (time * sweep_speed + notch_center))  # Sinusoidal sweep

        # Apply the phase shift to the corresponding frequency band
        band_range = (freqs >= notch_center - 0.05) & (freqs <= notch_center + 0.05)
        phase_img_np[band_range, :] += phase_shift

    # Ensure phase remains within the range [-π, π]
    phase_img_np = np.mod(phase_img_np + np.pi, 2 * np.pi) - np.pi

    # Convert back to uint8 for image processing
    phase_img_np = ((phase_img_np + np.pi) / (2 * np.pi) * 255).astype(np.uint8)

    return phase_img_np

def process_and_save_phased_image(output_filename, num_notches=6, depth=0.5, sweep_speed=0.1):
    # Load the combined image and split into magnitude and phase parts
    combined_img = Image.open('combined_image.png')
    combined_img_np = np.array(combined_img)
    height, width = combined_img_np.shape
    half_height = height // 2
    magnitude_img_np = combined_img_np[:half_height, :]
    phase_img_np = combined_img_np[half_height:, :]

    # Apply the phaser effect to the phase image
    phase_img_np = apply_phaser_effect(phase_img_np, num_notches, depth, sweep_speed)

    # Combine the modified phase image with the unchanged magnitude image
    new_combined_img = Image.new('L', (width, height))
    new_combined_img.paste(Image.fromarray(magnitude_img_np), (0, 0))
    new_combined_img.paste(Image.fromarray(phase_img_np), (0, half_height))
    new_combined_img.save(output_filename)

    # Load the new combined image and extract magnitude and phase
    combined_img_np = np.array(Image.open(output_filename))
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
    sf.write(f'reconstructed_audio_with_phaser.wav', y_reconstructed, sr)

    print(f"Reconstructed audio with phaser effect saved as {output_filename}.")

# Apply and save the phased image
process_and_save_phased_image('filtered_combined_image_phaser.png', num_notches=6, depth=0.5, sweep_speed=0.1)
