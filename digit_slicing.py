import numpy as np 
import cv2
import methods as meth
import grid_deletion as grid

'''
METHODS FOR SLICING A SUDOKU BOARD INTO DIGIT-SQUARES
'''
#Takes a Sudoku board with its grid deleted 
#Returns a list of tuples, (column, row, digit) where row and column
# are integers for the row/column location of the slice and digit is
# an np.array which is the image of the sliced area
#William
def slice_digits_from_gridfree_board(img: np.array) -> list:
    height, width = img.shape
    digit_array = list()
    #How large the digit slices will be, used in partitioning the picture
    column_segment = width//9
    row_segment = height//9
    #Loop through image, slicing and dicing
    for col in range(9):
        for row in range(9):
            digit_slice = get_digit_image(img, col, row, column_segment, row_segment)
            info_tup = (col, row, digit_slice)
            digit_array.append(info_tup)
    return digit_array


#Given a bunch of stuff, return the np.array of the desired slice
#William
def get_digit_image(img: np.array, col: int, row: int, column_segment: int, row_segment: int) -> np.array:
    #Dimensions for the slice
    row_start = row * row_segment
    row_end = row_start + row_segment
    col_start = col * column_segment
    col_end = col_start + column_segment
    #Now the actual slicing
    digit_img = img[row_start:row_end, col_start:col_end]
    return digit_img


#Put it all together: slice images from a raw Sudoku board
#William
def slice_board_to_digits(img: np.array, sensitivity: float, threshold: int) -> list:
    clear_img = grid.clear_grid(img, sensitivity, threshold)
    slice_list = slice_digits_from_gridfree_board(clear_img)
    return slice_list
