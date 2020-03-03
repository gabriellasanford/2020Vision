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
    edges = canny_edges(gray_img)
    # Detect points that form lines, and give them to calculate_cells() to get
    # info on the Sudoku grid.
    lines = cv2.HoughLinesP(edges, RHO, THETA, LINE_THRESH, MIN_LENGTH, MAX_GAP)
    grid_info = calculate_cells(lines)
    print(grid_info)


# Accepts a list of lines detected by the Hough transform, and returns a list
# with four elements, of the form: 
# [<top left x>, <top left y>, <cell width>, <cell height>]
def calculate_cells(hough_lines):
    grid_info = []
    x1_list, y1_list, x2_list, y2_list = decompose_lines(hough_lines)
    # Collect the top left corner of the grid
    start_x = min([min(x1_list), min(x2_list)])
    start_y = min([min(y1_list), min(y2_list)])
    grid_info.append(start_x)
    grid_info.append(start_y)
    # Collect the cell dimensions
    cell_tuple = cell_dims(x1_list, y1_list, x2_list, y2_list)
    grid_info.append(cell_tuple[0])
    grid_info.append(cell_tuple[1])
    return grid_info


# Helper function to calculate_cells(), factored out to improve readability.
# Accepts a list generated by cv2.HoughLinesP() and returns four distinct lists
# of points.
def decompose_lines(hough_lines):
    x1_list, y1_list, x2_list, y2_list = [], [], [], []
    # Split the hough line list into four lists of coordinates
    for line in hough_lines:
        for entry in line:
            x1_list.append(entry[0])
            y1_list.append(entry[1])
            x2_list.append(entry[2])
            y2_list.append(entry[3])
    return x1_list, y1_list, x2_list, y2_list


# Helper function to calculate_cells(), factored out to improve readability.
# Accepts four lists of numbers and returns a 2-tuple of the form:
# (<cell width>, <cell height>)
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
