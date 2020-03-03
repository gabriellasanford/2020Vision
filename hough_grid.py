import cv2
import numpy as np
import math
import matplotlib as plt


# Constants
# Canny thresholds
THRESH_ONE = 10
THRESH_TWO = 200

# HoughLinesP parameters
RHO = 1
THETA = math.pi/180
LINE_THRESH = 50
MIN_LENGTH = 25
MAX_GAP = 5

# Line drawing thickness
THICKNESS = 3

# Line drawing colors
BLACK = (255, 0, 0)


# Accepts a grayscale image and returns list of edges from Canny edge detector
def canny_edges(gray_img):
    return cv2.Canny(gray_img, THRESH_ONE, THRESH_TWO)


# Accepts a grayscale image and returns the approximate width and height of
# each cell in the Sudoku grid.
def count_sudoku(gray_img):
    cv2.imshow("Before", gray_img)
    edges = canny_edges(gray_img)
    # Detect points that form a line
    lines = cv2.HoughLinesP(edges, RHO, THETA, LINE_THRESH, MIN_LENGTH, MAX_GAP)
    print(calculate_cells(lines))


# Accepts a list of lines detected by the Hough transform, and returns a 2-tuple
# of the form: (<cell width>, <cell height>).
def calculate_cells(hough_lines):
    x1_list, y1_list, x2_list, y2_list = [], [], [], []
    # Split the hough line list into four lists of coordinates
    for line in hough_lines:
        for entry in line:
            x1_list.append(entry[0])
            y1_list.append(entry[1])
            x2_list.append(entry[2])
            y2_list.append(entry[3])
    return cell_dims(x1_list, y1_list, x2_list, y2_list)


# Helper function to calculate_cells(), factored out to improve readability
# Accepts four lists of numbers and returns a tuple
def cell_dims(x1_list, y1_list, x2_list, y2_list):
    # Calculate board width and height by taking the difference between max and
    # min x's and y's, across both lists of x's and y's
    width = ((max(x1_list) - min(x1_list)) \
        + (max(x2_list - min(x2_list)))) // 2
    height = ((max(y1_list) - min(y1_list)) \
        + (max(y2_list - min(y2_list)))) // 2
    # Calculate cell width and height
    cell_width = width // 9
    cell_height = height // 9
    return (cell_width, cell_height)
