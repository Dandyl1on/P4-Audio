import cv2
import numpy as np

Kernel = 1
Sigma = 2
Alpha = 1.5
Beta = -0.5
Gamma = 0

def sharpfunction(Kernel, Sigma, Alpha, Beta, Gamma):
    if Kernel % 2 == 0:
        Kernel += 1

    # Ændre magnitude_image.png til magnitude_image af de nye generet billeder
    image = cv2.imread('combined_image.png')

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get the dimensions of the grayscale image
    height, width = gray_image.shape

    # Split the image into top and bottom halves
    top_half = gray_image[:height // 2, :]
    bottom_half = gray_image[height // 2:, :]


    # gaussian kernel for sharpening
    gaussian_blur = cv2.GaussianBlur(top_half,(Kernel, 1),sigmaX=Sigma)


    # sharpening using addWeighted()
    sharpened_top_half = cv2.addWeighted(top_half,Alpha,gaussian_blur,Beta,Gamma)

    # Combine the sharpened top half and the unprocessed bottom half
    sharp1 = np.vstack((sharpened_top_half, bottom_half))

    cv2.imwrite("Sharpening.png", sharp1)

    # return width


    # cv2.imshow('sharp1', sharp1)
    # cv2.imshow('original', image)
    # cv2.waitKey(0)

# the addweight() method performs a linear comnbination of matrices
# which is simple arithmetic operations for example the first function will
# have the resultant matrix as image * 1.5 + gaussian_blur * (-0.5) + 0

# showing the images
