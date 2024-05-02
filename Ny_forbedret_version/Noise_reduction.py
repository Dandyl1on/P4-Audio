import numpy as np
from PIL import Image
import soundfile as sf
import librosa


def frequency_based_noise_reduction(image_path, reduction_factor, freq_ranges, sr, output_path):
    """
    Apply frequency-based noise reduction by attenuating specific frequency ranges in the magnitude image.

    Parameters:
    - image_path: Path to the input image (combined magnitude and phase).
    - reduction_factor: Factor by which to attenuate the unwanted frequencies (e.g., 0.5 to reduce by 50%).
    - freq_ranges: List of tuples specifying frequency ranges to attenuate [(low1, high1), (low2, high2)].
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

    # Apply noise reduction in the specified frequency ranges
    for low_freq, high_freq in freq_ranges:
        # Identify rows corresponding to the specified frequency range
        freq_mask = (freqs >= low_freq) & (freqs <= high_freq)

        # Attenuate the magnitudes in those rows by the reduction factor
        magnitude_img_np[freq_mask, :] *= reduction_factor

    # Convert back to uint8 while ensuring values are within the 0-255 range
    magnitude_img_np = np.clip(magnitude_img_np, 0, 255).astype(np.uint8)

    # Combine the modified magnitude image with the unchanged phase image
    new_combined_img = Image.new('L', (width, height))
    new_combined_img.paste(Image.fromarray(magnitude_img_np), (0, 0))
    new_combined_img.paste(Image.fromarray(phase_img_np), (0, half_height))
    new_combined_img.save(output_path)

    print(f"Noise-reduced image saved as {output_path}")


# Example usage:
image_path = 'combined_image.png'
output_path = 'frequency_based_noise_reduced_image.png'
reduction_factor = 0.5  # Attenuate by 50%
sr = 22050*2  # Sample rate of the original audio
freq_ranges = [(0, 300), (300, 500)]  # Low and high frequencies to reduce noise

frequency_based_noise_reduction(image_path, reduction_factor, freq_ranges, sr, output_path)
