import numpy as np
import librosa
from PIL import Image
import soundfile as sf
import os
from scipy.interpolate import interp1d

audio_path = "GI_GMF_B3_353_20140520_n.wav"
Path = "combined_image.png"

def infoFunc():

    global magnitude_db_max

    def reshape_to_custom(array, column_length):
        """ Reshape the array to have a fixed number of columns and calculate rows accordingly. """
        num_elements = array.size
        num_cols = column_length
        num_rows = int(np.ceil(num_elements / num_cols))
        resized_array = np.zeros((num_rows, num_cols))
        resized_array.flat[:num_elements] = array.flat
        return resized_array


    def apply_equal_loudness_contour(magnitude, sr, scale_factor=0.0):
        freqs = np.linspace(0, sr / 2, magnitude.shape[0])
        phon_20_curve = np.array([
            20.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0,
            2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0,
            11000.0, 12000.0, 13000.0, 14000.0, 15000.0, 16000.0, 17000.0, 18000.0,
            19000.0, 20000.0
        ])
        phon_20_db = np.array([
            60.0, 50.0, 40.0, 35.0, 30.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0,
            22.0, 21.0, 20.5, 20.0, 20.0, 20.0, 20.5, 21.0, 21.5,
            22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
            30.0, 32.0
        ])
        loudness_interpolator = interp1d(phon_20_curve, phon_20_db, kind='linear', fill_value="extrapolate")
        loudness_adjustments = loudness_interpolator(freqs)
        loudness_adjustments = loudness_adjustments * scale_factor
        loudness_adjustments = loudness_adjustments.reshape(-1, 1)
        magnitude_adjusted = magnitude * (10 ** (-loudness_adjustments / 20))
        return magnitude_adjusted


    def apply_inverse_equal_loudness_contour(magnitude, sr, scale_factor=0.0):
        freqs = np.linspace(0, sr / 2, magnitude.shape[0])
        phon_20_curve = np.array([
            20.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0,
            2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0,
            11000.0, 12000.0, 13000.0, 14000.0, 15000.0, 16000.0, 17000.0, 18000.0,
            19000.0, 20000.0
        ])
        phon_20_db = np.array([
            60.0, 50.0, 40.0, 35.0, 30.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0,
            22.0, 21.0, 20.5, 20.0, 20.0, 20.0, 20.5, 21.0, 21.5,
            22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
            30.0, 32.0
        ])
        loudness_interpolator = interp1d(phon_20_curve, phon_20_db, kind='linear', fill_value="extrapolate")
        loudness_adjustments = loudness_interpolator(freqs)
        loudness_adjustments = loudness_adjustments * scale_factor
        loudness_adjustments = loudness_adjustments.reshape(-1, 1)
        magnitude_adjusted = magnitude / (10 ** (-loudness_adjustments / 20))
        return magnitude_adjusted


    def save_combined_image(magnitude_normalized, phase_normalized, output_path='combined_image.png'):
        # Convert arrays to images
        magnitude_image = Image.fromarray((magnitude_normalized * 255).astype(np.uint8), mode='L')
        phase_image = Image.fromarray((phase_normalized * 255).astype(np.uint8), mode='L')

        # Stack the images and save the combined image
        combined_img = Image.new('L', (magnitude_image.width, magnitude_image.height + phase_image.height))
        combined_img.paste(magnitude_image, (0, 0))
        combined_img.paste(phase_image, (0, magnitude_image.height))
        combined_img.save(output_path)
        print(f"Combined image saved to {output_path}")


    def construct_image_from_audio(audio_path, output_image_path='combined_image.png'):
        # Load audio file and compute STFT
        y, sr = librosa.load(audio_path, sr=None)# sr = None
        D = librosa.stft(y)
        magnitude = np.abs(D)
        phase = np.angle(D)

        # Calculate the column length as twice the square root of the array length
        column_length = int(1.5 * np.ceil(np.sqrt(magnitude.size)))

        # Reshape magnitude and phase arrays
        magnitude_resized = reshape_to_custom(magnitude, column_length)
        phase_resized = reshape_to_custom(phase, column_length)

        # Apply equal-loudness contour before normalization
        magnitude = apply_equal_loudness_contour(magnitude_resized, sr, scale_factor=1.0)

        # Normalize magnitude to a logarithmic scale and then to [0, 1]
        magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))
        magnitude_db_min = np.min(magnitude_db)
        magnitude_db_max = np.max(magnitude_db)
        magnitude_normalized = (magnitude_db - magnitude_db_min) / (magnitude_db_max - magnitude_db_min)

        # Normalize phase to [0, 1]
        phase_normalized = (phase_resized + np.pi) / (2 * np.pi)

        # Save the combined image
        save_combined_image(magnitude_normalized, phase_normalized, output_image_path)

        return magnitude_db_max, magnitude_db_min, sr, D


    def reconstruct_audio_from_image(magnitude_db_max, magnitude_db_min, sr, D, combined_image_path='combined_image.png'):
        # Load combined image and extract magnitude and phase
        combined_img_np = np.array(Image.open(combined_image_path))
        img_height = combined_img_np.shape[0] // 2
        magnitude_img_np = combined_img_np[:img_height, :]
        phase_img_np = combined_img_np[img_height:, :]

        # Normalize and denormalize magnitude and phase
        magnitude_normalized = magnitude_img_np / 255.0
        phase_normalized = phase_img_np / 255.0

        # Denormalize magnitude
        magnitude_db = magnitude_normalized * (magnitude_db_max - magnitude_db_min) + magnitude_db_min
        magnitude = 10 ** (magnitude_db / 20)  # Convert dB  back to amplitude

        # Apply the inverse equal-loudness contour
        magnitude = apply_inverse_equal_loudness_contour(magnitude, sr, scale_factor=1.0)

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
        output_audio_path = 'reconstructed_audio_from_combined_image.wav'
        sf.write(output_audio_path, y_reconstructed, sr)
        print(f"Reconstructed audio saved to {output_audio_path}")

    magnitude_db_max, magnitude_db_min, sr, D = construct_image_from_audio(audio_path)
    reconstruct_audio_from_image(magnitude_db_max, magnitude_db_min, sr, D, Path)
    return magnitude_db_max, magnitude_db_min, sr, D

