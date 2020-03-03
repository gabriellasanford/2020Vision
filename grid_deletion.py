import numpy as np
import cv2
import methods as meth


'''
METHODS FOR ISOLATING CELLS
'''
#Deletes any rows or columns that are predominantely black
#Useful?  Perhaps because then segmenting the Sudoku board will be easier
#Black = 0, white = 255
#William
def delete_grid(img: np.array, threshold: int) -> np.array:
    rows, columns = img.shape
    print("Rows: " + str(rows))
    print("Cols: " + str(columns))
    columns_to_kill = list()
    rows_to_kill = list()
    #Loop through rows, looking for ones to kill
    for r in range(rows):
        counter = 0
        for c in range(columns):
            if img[r][c] < threshold:
                counter += 1
        if counter/rows > .15:
            rows_to_kill.append(r)
    #Same thing for columns
    for c in range(columns):
        counter = 0
        for r in range(rows):
            if img[r][c] < threshold:
                counter += 1
        if counter/rows > .15:
            columns_to_kill.append(c)
    print(rows_to_kill)
    print(columns_to_kill)
    #Now reverse these boys so we don't get out of bounds execeptions
    rows_to_kill.reverse()
    columns_to_kill.reverse()
    #NOW KILL IT WITH FIRE
    for r in rows_to_kill:
        img = np.delete(img, r, 0)
    for c in columns_to_kill:
        img = np.delete(img, c, 1)
    return img

'''
Grid deletion via summation and then masking
William
'''
#Perhaps instead of doing the above, would summing work?  The rows/columns would have a high value if summed...
#Then just delete the rows/columns that have a sum that's super high/low depending on the value for white and black
def sum_grid_kill(img: np.array, sensitivity: float) -> np.array:
    #Get the shape of our image
    rows, columns = img.shape
    #Make arrays with the sum of the pixel values for the columns and rows respectively
    column_sum = np.sum(img, 0)
    row_sum = np.sum(img, 1)
    #Find rows and columns with a low sum-value, which means there are a lot of black pixels 
    columns_to_kill = list()
    rows_to_kill = list()
    col_avg = np.average(column_sum)
    row_avg = np.average(row_sum)
    for i in range(len(column_sum)):
        #If there are a lot of black pixels, add it to the kill list
        if column_sum[i] < (col_avg / sensitivity):
            columns_to_kill.append(i)
    for i in range(len(row_sum)):
        #If there are a lot of black pixels, add it to the kill list
        if row_sum[i] < (row_avg / sensitivity):
            rows_to_kill.append(i)
    #Reverse the lists so we can kill the rows/columns without messing up our bounds
    columns_to_kill.reverse()
    rows_to_kill.reverse()
    #KILL IT WITH FIRE
    for col_num in columns_to_kill:
        img = np.delete(img, col_num, 1)
    for row_num in rows_to_kill:
        img = np.delete(img, row_num, 0)
    return img

#Masks anything about threshold to white
#Remember that black = 0, white = 255
def mask_gray_away(img: np.array, threshold: int) -> np.array:
    img_copy = img.copy()
    img_copy[img_copy > threshold] = 255
    return img_copy

#combines sum_grid_kill and mask_gray_away into a single method
#img: np.array which is the image of the Sudoku board
#sensitivity: how strict to be in getting rid of rows/columns with lots of black
#threshold: the masking threshold for anything about threshold gets sent to white
def clear_grid(img:np.array, sensitivity: float, threshold: int) -> np.array:
    img = sum_grid_kill(img, sensitivity)
    img = mask_gray_away(img, threshold)
    return img

'''
End of grid deletion
'''



'''
End of cell isolation
'''