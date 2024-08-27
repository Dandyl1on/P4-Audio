import numpy as np
from PIL import Image
import soundfile as sf
import librosa


def apply_equalization(image_path, eq_settings, sr, output_path):
    """
    Apply frequency-based equalization by adjusting the gain of specific frequency bands in the magnitude image.

    Parameters:
    - image_path: Path to the input image (combined magnitude and phase).
    - eq_settings: Dictionary specifying gain settings for frequency ranges, e.g., { (low_freq, high_freq): gain_factor }.
    - sr: Sample rate of the audio signal (needed to map frequency ranges).
    - output_path: Path to save the output image.
    """
    # Load the combined image
    combined_img = Image.open(image_path)
    combined_img_np = np.array(combined_img)
    height, width = combined_img_np.shape
    half_height = height // 2

    # Separate magnitude and phase
    magnitude_img_np = combined_img_np[:half_height, :].astype(np.float64)  # Convert to float for processing
    phase_img_np = combined_img_np[half_height:, :]

    # Calculate the frequency for each row in the magnitude image
    freqs = np.linspace(0, sr / 2, half_height)

    # Apply equalization based on the specified settings
    for (low_freq, high_freq), gain_factor in eq_settings.items():
        # Identify rows corresponding to the specified frequency range
        freq_mask = (freqs >= low_freq) & (freqs <= high_freq)

        # Apply the gain to those frequencies
        magnitude_img_np[freq_mask, :] *= gain_factor

    # Convert back to uint8 while ensuring values are within the 0-255 range
    magnitude_img_np = np.clip(magnitude_img_np, 0, 255).astype(np.uint8)

    # Combine the modified magnitude image with the unchanged phase image
    new_combined_img = Image.new('L', (width, height))
    new_combined_img.paste(Image.fromarray(magnitude_img_np), (0, 0))
    new_combined_img.paste(Image.fromarray(phase_img_np), (0, half_height))
    new_combined_img.save(output_path)

    print(f"Equalized image saved as {output_path}")


# Example usage:
image_path = 'combined_image.png'
output_path = 'equalized_image.png'
sr = 22050  # Sample rate of the original audio

# Define the frequency bands and corresponding gain factors
eq_settings = {
    (0, 200): 0.7,  # Boost low frequencies (bass)
    (200, 2000): 0.9,  # Leave mid frequencies unchanged
    (2000, 5000): 1.4,  # Reduce upper mids
    (5000, 11025): 1.3  # Slightly boost high frequencies (treble)
}

apply_equalization(image_path, eq_settings, sr, output_path)
