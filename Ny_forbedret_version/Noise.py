import cv2
import numpy as np


def add_gaussian_noise(image, mean=15, sigma=25):
    """
    Add Gaussian noise to the top half of an image and make it grayscale.

    :param image: Input image
    :param mean: Mean of the Gaussian noise
    :param sigma: Standard deviation of the Gaussian noise
    :return: Noisy image
    """
    # Convert image to float32
    image = image.astype(np.float32)

    # Generate Gaussian noise for the grayscale image
    noise = np.random.normal(mean, sigma, image.shape[:2])

    # Create a mask for the top half
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    mask[:height // 2] = 1  # Mask for top half

    # Apply noise only to the top half of the image
    noisy_image = image.copy()
    noisy_image += mask[:, :, np.newaxis] * noise[:, :, np.newaxis]

    # Clip the pixel values to stay within the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)

    # Convert image back to uint8
    return noisy_image.astype(np.uint8)


# Load the image
image_path = 'combined_image.png'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Ensure the image is in RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Add Gaussian noise to the top half
noisy_image = add_gaussian_noise(image)

# Convert noisy image to grayscale
noisy_image_gray = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2GRAY)

# Save the noisy image as PNG
cv2.imwrite('noisy_image.png', cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR))
cv2.imwrite('noisy_image_gray.png', noisy_image_gray)

# Display the images
cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image', noisy_image)
cv2.imshow('Noisy Image (Gray)', noisy_image_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
