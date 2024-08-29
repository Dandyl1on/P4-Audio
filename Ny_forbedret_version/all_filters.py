import numpy as np
import librosa
from PIL import Image
import soundfile as sf
from main import mainfunc

magnitude_db_max, magnitude_db_min, sr, D = mainfunc()

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

def process_and_save_filtered_image(filter_type, cutoff_freqs, output_filename):
    # Load the combined image and split into magnitude and phase parts
    combined_img = Image.open('combined_image.png')
    combined_img_np = np.array(combined_img)
    height, width = combined_img_np.shape
    half_height = height // 2
    magnitude_img_np = combined_img_np[:half_height, :]
    phase_img_np = combined_img_np[half_height:, :]

    # Create the filter mask
    mask = create_filter_mask(magnitude_img_np.shape, sr, filter_type, cutoff_freqs)

    # Apply the mask to the magnitude image
    filtered_magnitude_img_np = magnitude_img_np * mask

    # Combine the filtered magnitude image with the unchanged phase image
    new_combined_img = Image.new('L', (width, height))
    new_combined_img.paste(Image.fromarray(filtered_magnitude_img_np), (0, 0))
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
    sf.write(f'reconstructed_audio_from_{filter_type}_filter.wav', y_reconstructed, sr)

    print(f"Reconstructed audio with {filter_type} filter saved.")

# Parameters
sr = 22050
low_freq = 10000  # Low cutoff frequency in Hz
high_freq = 13000  # High cutoff frequency in Hz
notch_freq = 12000  # Notch frequency in Hz
bandwidth = 100  # Bandwidth of the notch filter in Hz

def ApplyFilters():

    print(high_freq)
    print(low_freq)
    print(notch_freq)
    print(bandwidth)

    # Apply filters and save results
    process_and_save_filtered_image('bandpass', (low_freq, high_freq), 'filtered_combined_image_bandpass.png')
    process_and_save_filtered_image('lowpass', (high_freq,), 'filtered_combined_image_lowpass.png')
    process_and_save_filtered_image('highpass', (low_freq,), 'filtered_combined_image_highpass.png')
    process_and_save_filtered_image('notch', (notch_freq, bandwidth), 'filtered_combined_image_notch.png')
