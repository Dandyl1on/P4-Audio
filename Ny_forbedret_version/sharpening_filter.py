import cv2
import numpy as np

def sharpfunction():
    global sharp1
    # Ændre magnitude_image.png til magnitude_image af de nye generet billeder
    image = cv2.imread('magnitude_image.png')

    # gaussian kernel for sharpening
    gaussian_blur = cv2.GaussianBlur(image,(7,7),sigmaX=2)

    # sharpening using addWeighted()
    sharp1 = cv2.addWeighted(image,1.5,gaussian_blur,-0.5,0)

    cv2.imwrite("Sharpening.png", sharp1)

    return sharp1

    # cv2.imshow('sharp1', sharp1)
    # cv2.imshow('original', image)
    # cv2.waitKey(0)

# the addweight() method performs a linear comnbination of matrices
# which is simple arithmetic operations for example the first function will
# have the resultant matrix as image * 1.5 + gaussian_blur * (-0.5) + 0

# showing the images
