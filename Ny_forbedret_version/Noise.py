import cv2
import numpy as np

mean = 0
sigma = 0

def add_gaussian_noise(mean, sigma):
    """
    Add Gaussian noise to the top half of an image and make it grayscale.

    :param image: Input image
    :param mean: Mean of the Gaussian noise
    :param sigma: Standard deviation of the Gaussian noise
    :return: Noisy image
    """
    image = cv2.imread('combined_image.png')
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
    noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2GRAY)

    # Convert image back to uint8

    cv2.imwrite('Noise.png', noisy_image)

    return noisy_image

# # Ensure the image is in RGB format
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # Add Gaussian noise to the top half
# noisy_image = add_gaussian_noise(image)
#
# # Convert noisy image to grayscale
#
#
# # Save the noisy image as PNG
# cv2.imwrite('noisy_image.png', cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR))
#
#
# # Display the images
# cv2.imshow('Original Image', image)
# cv2.imshow('Noisy Image', noisy_image)
# cv2.imshow('Noisy Image (Gray)', noisy_image_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
