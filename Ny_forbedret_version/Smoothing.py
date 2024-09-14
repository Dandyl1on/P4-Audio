import cv2
import numpy as np

kernel = 0
weightval = 0

def apply_smoothing(kernel, weightval):
    """
    Apply smoothing to the top half of an image with adjustable intensity.

    :param image: Input image
    :param kernel_size: Size of the smoothing kernel
    :param weight: Weight for blending the smoothed image
    :return: Smoothed image
    """
    image = cv2.imread('combined_image.png')
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
    smoothed_top_half = cv2.GaussianBlur(top_half, (kernel, 1), 0)

    # Blend the smoothed top half with the original top half
    smoothed_top_half = cv2.addWeighted(top_half, 1 - weightval, smoothed_top_half, weightval, 0)

    # Place the smoothed top half back into the image
    smoothed_image[:height // 2] = smoothed_top_half

    # Convert smoothed image to grayscale
    smoothed_image_gray = cv2.cvtColor(smoothed_image, cv2.COLOR_RGB2GRAY)

    cv2.imwrite('Smoothing.png', smoothed_image_gray)

    return smoothed_image_gray


# Load the image

# Save the smoothed image as PNG


# Display the images
# cv2.imshow('Original Image', image)
# cv2.imshow('Smoothed Image', smoothed_image)
# cv2.imshow('Smoothed Image (Gray)', smoothed_image_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
