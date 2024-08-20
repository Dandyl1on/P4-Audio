import numpy as np
import librosa
from PIL import Image
import soundfile as sf
from main import magnitude_db_max, magnitude_db_min, sr, D

def apply_spectral_inversion(magnitude_img_np):
    """
    Apply spectral inversion by swapping the top and bottom halves of the magnitude image.
    Handles cases where the number of rows is odd by adding padding if necessary.
    """
    rows, cols = magnitude_img_np.shape
    half_rows = rows // 2

    # Create a new array to handle the spectral inversion
    inverted_magnitude_img_np = np.zeros_like(magnitude_img_np)

    if rows % 2 == 0:
        # If rows are even, simply swap the halves
        inverted_magnitude_img_np[:half_rows, :] = magnitude_img_np[half_rows:, :]
        inverted_magnitude_img_np[half_rows:, :] = magnitude_img_np[:half_rows, :]
    else:
        # If rows are odd, pad the image to make rows even
        padded_magnitude_img_np = np.pad(magnitude_img_np, ((0, 1), (0, 0)), mode='constant', constant_values=0)
        padded_rows, padded_cols = padded_magnitude_img_np.shape
        half_padded_rows = padded_rows // 2

        # Apply spectral inversion on padded image
        inverted_padded_magnitude_img_np = np.zeros_like(padded_magnitude_img_np)
        inverted_padded_magnitude_img_np[:half_padded_rows, :] = padded_magnitude_img_np[half_padded_rows:, :]
        inverted_padded_magnitude_img_np[half_padded_rows:, :] = padded_magnitude_img_np[:half_padded_rows, :]

        # Remove padding
        inverted_magnitude_img_np = inverted_padded_magnitude_img_np[:-1, :]

    return inverted_magnitude_img_np

def process_and_save_spectral_inversion_image(output_filename):
    """
    Process the image to apply spectral inversion and save the result.
    """
    # Load the combined image and split into magnitude and phase parts
    combined_img = Image.open('combined_image.png')
    combined_img_np = np.array(combined_img)
    height, width = combined_img_np.shape
    half_height = height // 2
    magnitude_img_np = combined_img_np[:half_height, :]
    phase_img_np = combined_img_np[half_height:, :]

    # Apply spectral inversion to the magnitude image
    magnitude_img_np = apply_spectral_inversion(magnitude_img_np)

    # Combine the modified magnitude image with the unchanged phase image
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
    sf.write(f'reconstructed_audio_with_spectral_inversion.wav', y_reconstructed, sr)

    print(f"Reconstructed audio with spectral inversion saved as {output_filename}.")

# Parameters
sr = 44100  # Sample rate

# Apply spectral inversion and save results
process_and_save_spectral_inversion_image('filtered_combined_image_spectral_inversion.png')
