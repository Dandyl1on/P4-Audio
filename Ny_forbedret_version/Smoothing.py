import cv2
import numpy as np


def apply_smoothing(image, kernel_size=(3, 3), weight=0.5):
    """
    Apply smoothing to the top half of an image with adjustable intensity.

    :param image: Input image
    :param kernel_size: Size of the smoothing kernel
    :param weight: Weight for blending the smoothed image
    :return: Smoothed image
    """
    # Convert image to float32
    image = image.astype(np.float32)

    # Create a mask for the top half
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:height // 2] = 1  # Mask for top half

    # Apply smoothing only to the top half of the image
    smoothed_image = image.copy()
    top_half = image[:height // 2]

    # Apply Gaussian blur to the top half
    smoothed_top_half = cv2.GaussianBlur(top_half, kernel_size, 0)

    # Blend the smoothed top half with the original top half
    smoothed_top_half = cv2.addWeighted(top_half, 1 - weight, smoothed_top_half, weight, 0)

    # Place the smoothed top half back into the image
    smoothed_image[:height // 2] = smoothed_top_half

    return smoothed_image.astype(np.uint8)


# Load the image
image_path = 'combined_image.png'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Ensure the image is in RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply smoothing to the top half with adjusted intensity
smoothed_image = apply_smoothing(image, kernel_size=(15, 15), weight=0.4)

# Convert smoothed image to grayscale
smoothed_image_gray = cv2.cvtColor(smoothed_image, cv2.COLOR_RGB2GRAY)

# Save the smoothed image as PNG
cv2.imwrite('smoothed_image.png', cv2.cvtColor(smoothed_image, cv2.COLOR_RGB2BGR))
cv2.imwrite('smoothed_image_gray.png', smoothed_image_gray)

# Display the images
cv2.imshow('Original Image', image)
cv2.imshow('Smoothed Image', smoothed_image)
cv2.imshow('Smoothed Image (Gray)', smoothed_image_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
