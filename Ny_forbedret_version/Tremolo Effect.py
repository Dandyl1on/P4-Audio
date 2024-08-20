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
        mask = np.array([[0 if (notch_freq - bandwidth / 2) <= freq <= (notch_freq + bandwidth / 2) else 1 for _ in range(cols)] for freq in freqs])

    return mask

def apply_tremolo(magnitude_img_np, rate=5, depth=0.5):
    """
    Apply tremolo to the magnitude image by modulating amplitude over time.
    - rate: The frequency of the tremolo effect (in Hz).
    - depth: The intensity of the modulation (0.0 to 1.0).
    """
    rows, cols = magnitude_img_np.shape
    time = np.linspace(0, cols / sr, cols)
    tremolo = 1 + depth * np.sin(2 * np.pi * rate * time)  # Tremolo waveform

    # Apply tremolo along the time axis (columns)
    modulated_magnitude = magnitude_img_np * tremolo

    # Ensure values stay within the valid range [0, 255]
    modulated_magnitude = np.clip(modulated_magnitude, 0, 255).astype(np.uint8)
    return modulated_magnitude

def apply_amplitude_modulation(magnitude_img_np, carrier_freq=10):
    """
    Apply amplitude modulation to the magnitude image.
    - carrier_freq: The frequency of the AM carrier signal (in Hz).
    """
    rows, cols = magnitude_img_np.shape
    time = np.linspace(0, cols / sr, cols)
    carrier_wave = 0.5 * (1 + np.sin(2 * np.pi * carrier_freq * time))  # AM carrier waveform

    # Apply amplitude modulation along the time axis (columns)
    modulated_magnitude = magnitude_img_np * carrier_wave

    # Ensure values stay within the valid range [0, 255]
    modulated_magnitude = np.clip(modulated_magnitude, 0, 255).astype(np.uint8)
    return modulated_magnitude

def process_and_save_effect_image(effect_type, output_filename, **effect_params):
    # Load the combined image and split into magnitude and phase parts
    combined_img = Image.open('combined_image.png')
    combined_img_np = np.array(combined_img)
    height, width = combined_img_np.shape
    half_height = height // 2
    magnitude_img_np = combined_img_np[:half_height, :]
    phase_img_np = combined_img_np[half_height:, :]

    # Apply the selected effect to the magnitude image
    if effect_type == 'tremolo':
        magnitude_img_np = apply_tremolo(magnitude_img_np, **effect_params)
    elif effect_type == 'amplitude_modulation':
        magnitude_img_np = apply_amplitude_modulation(magnitude_img_np, **effect_params)

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
    sf.write(f'reconstructed_audio_with_{effect_type}.wav', y_reconstructed, sr)

    print(f"Reconstructed audio with {effect_type} saved as {output_filename}.")

# Parameters
sr = 22050  # Sample rate
tremolo_rate = 5  # Tremolo rate in Hz
tremolo_depth = 0.5  # Tremolo depth (0.0 to 1.0)
am_carrier_freq = 10  # Carrier frequency for amplitude modulation in Hz

# Apply effects and save results
process_and_save_effect_image('tremolo', 'filtered_combined_image_tremolo.png', rate=tremolo_rate, depth=tremolo_depth)
process_and_save_effect_image('amplitude_modulation', 'filtered_combined_image_amplitude_modulation.png', carrier_freq=am_carrier_freq)
