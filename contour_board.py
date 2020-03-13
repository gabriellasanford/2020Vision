import cv2
import hough_grid as hough
import numpy as np


# Constants
# File path of image to be tested.
IMG_PATH = "sudoku_midterm/sudoku5.png"

# Values for thresholding.
WHITE = 255
NEIGHBORHOOD_SIZE = 7
C = 15


# Accepts an image and returns the thresholded image.
def threshold(img):
    thresh_img = cv2.adaptiveThreshold(img, WHITE,\
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,\
        NEIGHBORHOOD_SIZE, C)
    return thresh_img


# Accepts a grayscale image and returns image with its histogram equalized.
def equalize(img):
    return cv2.equalizeHist(img)


# Accepts an image, finds the contours using cv2's functions, and returns both
# a list of contours and a new image with the contours drawn on.
def contours(img):
    img = threshold(img)
    shapes, hierarchy = cv2.findContours(img, cv2.RETR_TREE,\
        cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, shapes, -1, 0)
    return shapes, img


# Accepts an image and a string and displays the image to the screen.
def display(title, img):
    cv2.imshow(title, img)
    cv2.waitKey()


# Used for testing
def main():
    image = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    display("Original", image)
    sharp_img = hough.unsharp_mask(image)
    display("Sharpened", sharp_img)
    shapes, contour_img = contours(sharp_img)
    display("Contoured", contour_img)


if __name__ == "__main__":
    main()