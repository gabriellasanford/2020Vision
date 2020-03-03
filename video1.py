import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import knnClassify as knn
import math
import methods as meth
import hough_grid as hough
'''
OLD STUFF, should be refactored
AKA Dr. Hochberg's stuff
'''

# stores a trained knnClassify. Initiated as None to avoid
# unnecessary overhead if not used
classifier = None

img_orig = cv2.imread("images/sudoku1.png", cv2.IMREAD_GRAYSCALE)

hough.detect(img_orig)

# Show the original image
# This is a matplotlib display, so we must close the window to move forward
#plt.imshow(255-img_orig, cmap=plt.cm.binary)
#plt.show()

element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7), (3, 3))
img = 255-cv2.dilate(255-img_orig, element)
# This is a matplotlib display, so we must close the window to move forward
#plt.imshow(255-img, cmap=plt.cm.binary)
#plt.show()

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Filter by Inertia
#params.filterByInertia = True
#params.minInertiaRatio = 0.2

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)
 
# Detect blobs.
keypoints = detector.detect(img)
#print([(k.pt, k.size) for k in keypoints])

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the
# size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(img, keypoints, None, (255,0,255),\
                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

###
### Here we start using imshow() --- the native OpenCV image viewer
### These windows do not lock up the main thread
###
'''
End of old block
'''


'''
IMAGE-TO-BOARD block
'''
def make_classifier(feature):
    print("Training, please wait. . .")
    training_map = {}

    fonts = [cv2.FONT_HERSHEY_SIMPLEX,\
             cv2.FONT_HERSHEY_DUPLEX,cv2.FONT_HERSHEY_COMPLEX,\
             cv2.FONT_HERSHEY_TRIPLEX,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,\
             cv2.FONT_HERSHEY_SCRIPT_COMPLEX,cv2.FONT_ITALIC]
    
    for i in range(0,10):
        training_map[i] = []
        for k in range(0,50):
            img = np.zeros((28, 28,3), dtype=np.uint8)
            img = cv2.putText(img, str(i), (5+k%2,22+k%3), fonts[k%len(fonts)],\
                              0.85, (255,255,255), 2, cv2.LINE_AA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #img = cv2.bitwise_not(img)
            training_map[i].append(img)
            """cv2.imshow("digit"+str(i), img)
            while cv2.waitKey(1) & 0xFF != ord('n'):
                continue"""

    classifier = knn.make_trained_knn(feature, training_map)
    knn.test_existing_knn(meth.slantiness, classifier, training_map)
    return classifier


# Sri, Anthony, Eniola
# takes an np.array of digit images and returns an np.array of their respective digits
def classify_imgs(digit_imgs):
    feature = meth.slantiness
    
    digit_imgs = np.array(list(map(lambda i: cv2.resize(i, (28, 28)), digit_imgs)))
    global classifier
    if classifier is None:
        classifier = make_classifier(feature)

    result = []
    for digit in digit_imgs:
        img = cv2.resize(digit, (28, 28))
        img = cv2.bitwise_not(img)
        predict = knn.classify_digit(classifier, np.array(feature(img)))
        result.append(predict)
        #print(predict)
        #print(img)
        #cv2.imshow("One Digit", img)
        #while cv2.waitKey(1) & 0xFF != ord('n'):
        #    continue
    
    return np.array(result)
    

def classify_single_img(digit_img):
    feature = meth.slantiness

    digit_img = cv2.resize(digit_img, (28,28))
    
    global classifier
    if classifier is None:
        classifier = make_classifier(feature)

    digit_img = cv2.bitwise_not(digit_img)
    prediction = knn.classify_digit(classifier, np.array(feature(digit_img)))

    return prediction    


#Gets the position tuple for a keypoints and returns it
def get_position(key_point: cv2.KeyPoint) -> tuple:
    size = int(key_point.size)
    position = tuple(int(x-size/2) for x in key_point.pt)
    return position


#Returns the y position of the keypoint
def get_y_position(key_point: cv2.KeyPoint) -> int:
    return get_position(key_point)[1]


#Returns the x position of the keypoint
def get_x_position(key_point: cv2.KeyPoint) -> int:
    return get_position(key_point)[0]


#Checks to see if two values are within a particular range of each other
def within_spread(a: int, b: int, spread: int) -> bool:
    return abs(a - b) <= spread


#William's method for getting row/column position of keypoints
#Takes a list of keypoints, a spread for variance in keypoint location, and "x" or "y" 
# to denote whether this is for columns or rows respectively.
#Returns a dictionary mapping the keypoints' x/y values to their column/row position
def map_keypoints(list_of_points: list, spread: int, axis: str) -> dict:
    #Sort the keypoints by (x, y) position
    if axis == "x":
        list_of_points.sort(key=get_x_position)
    elif axis == "y":
        list_of_points.sort(key=get_y_position)
    #Make dictionaries to hold row and column mappings
    dictionary = dict()
    #Non-sense starter values 
    last_val = -5 * spread
    row_or_column = -1
    for k in list_of_points:
        position = get_position(k)
        if axis == "x":
            current_val = position[0]
        elif axis == "y":
            current_val = position[1]
        #Now check the current values against the past values, if they're different *enough* then 
        #update our row or column position
        if current_val is not last_val:
            #Values didn't match:
            # If they're close enough, then they get the same value
            # If they're not close enough, then update what column we're in and add it
            if within_spread(current_val, last_val, spread):
                dictionary[current_val] = row_or_column
            else:
                row_or_column += 1
                dictionary[current_val] = row_or_column

        #Update our checking values!
        last_val = current_val

    return dictionary


#Method for making a 2-d list for a Sudoku board
#Takes keypoints, classifies the digits puts them in the appropriate spot 
def keypoints_to_board(list_of_points: list, x_spread: int, y_spread: int):
    #Make row and column dictionaries 
    row_dict = map_keypoints(list_of_points, x_spread, "y")
    column_dict = map_keypoints(list_of_points, y_spread, "x")
    #Keep for now until this works with more than one board
    '''
    print("rows:")
    print(row_dict)
    print("columns:")
    print(column_dict)
    '''
    #Construct the Sudoku board as 2-d list, fill it with -1s as empty values
    board = [[0 for col in range(9)] for row in range(9)]

    #Loop through and get the images to classify
    #Classify them, and put them in their spot on the board
    for k in keypoints:
        #Get the image of the digit from the keypoint
        digit_img = get_digit_image(k)
        #Classify the digit in the image
        digit_val = int(classify_single_img(digit_img))
        #Get the position of the digit, then convert that to it's row and column position

        #This is where the bug is, 
         #   which is odd since the x,y coords are coming from the same function
          #  as the one that made the dictionary in the first place...

        x = get_x_position(k)
        #print("x: " + str(x))
        y = get_y_position(k)
        #print("y:" + str(y))
        column = column_dict.get(x)
        #print("col: " + str(column))
        row = row_dict.get(y)
        #print("row: " + str(row))
        
        for key in column_dict.keys():
            x = get_x_position(k)
            if within_spread(x, key, 15):
                #print("we out here")
                column = column_dict.get(key)
                break

        for key in row_dict.keys():
            y = get_y_position(k)
            #print("Inner y: " + str(y))
            #print(key)
            if within_spread(y, key, 15):
                row = row_dict.get(key)
                #print("we out here x2")
                break
        #print("col: " + str(column))
        #print("row: "+ str(row))
        #Add the value of the digit to the Sudoku board
        if row == None : row = 0        # Catch error
        if column == None : column = 0  # Catch error
        board[row][column] = digit_val

    return board


#Returns the digit image of a keypoint from a Sudoku board
#Code courtesy of Dr. Hochberg for snipping the blobs
def get_digit_image(k_point: cv2.KeyPoint) -> np.array:
    size = int(k_point.size)
    p = tuple(int(x-size/2) for x in k_point.pt)
    cv2.rectangle(img, p, (p[0]+size, p[1]+size), 200, 5)
    digit_img = img_orig[p[1]:p[1]+size, p[0]:p[0]+size]
    return digit_img


#Takes an image of a Sudoku board, returns the Sudoku board as a 2-d list
def sudoku_image_to_board(image: np.array):
    dimensions = image.shape
    #Divide by 18 since that's 9 * 2, which is each row/column's half height/width 
    y_spread = dimensions[0] // 18
    x_spread = dimensions[1] // 18
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7), (3, 3))
    #So we can dilate ourselves into a problem...
    img = 255-cv2.dilate(255-image, element)
    cv2.imshow("test", img)
    cv2.waitKey()
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)
    # Detect blobs.
    keypoints = detector.detect(img)
    #Make the board 
    board = keypoints_to_board(keypoints, x_spread, y_spread)
    return board

'''
End of image-to-board block
'''

'''
METHODS FOR ISOLATING CELLS
'''
#Deletes any rows or columns that are predominantely black
#Useful?  Perhaps because then segmenting the Sudoku board will be easier
#Black = 0, white = 255
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
End of cell isolation
'''

'''
GROUP SUBMISSIONS
'''

# deduce what sudoku cell each keypoint is in.
# returns a list of tuples (i, j) where is i is
# the row index and j is the column index.
def duy_paul_gabriella_keypoints_to_cells(keypoint_list):
    left = sys.maxsize
    right = 0
    top = sys.maxsize
    bottom = 0
    cells = []
    for a_keypoint in keypoint_list:
        left = min(left, a_keypoint.pt[0])
        right = max(right, a_keypoint.pt[0])
        top = min(top, a_keypoint.pt[1])
        bottom = max(bottom, a_keypoint.pt[1])
    # distance from one column center to the next
    x_bucket_size = (right - left) / 8
    # distance from one row center to the next
    y_bucket_size = (bottom - top) / 8
    for a_keypoint in keypoint_list:
        x, y = a_keypoint.pt
        j = round((x - left) / x_bucket_size)
        i = round((y - top) / y_bucket_size)
        cells.append((i, j))
    return cells


# #Matt & Michael 
# #map keypoints x, y to a grid coordinate

# #Find interval of x by dividing  the difference of min and max
# maxX = max(k.pt[0] for k in keypoints)
# minX = min(k.pt[0] for k in keypoints)
# intervalX = (maxX-minX)//8

# #Find interval of x by dividing  the difference of min and max
# maxY = max(k.pt[1] for k in keypoints)
# minY = min(k.pt[1] for k in keypoints)
# intervalY = (maxY-minY)//8

#function to return grid coor from keypoint x,y
def pos_abs_to_grid(k_point):
    return(k_point[0]//intervalX,k_point[1]//intervalY)

#function that takes a list of k and returns an list of coords
def ks_to_coords():
    coords = [pos_abs_to_grid(k.pt) for k in keypoints]
    return coords

#function that replaces k_point[x,y] with the cooridantes
def ks_replace_coords():
    for k in keypoints:
        k.pt = (pos_abs_to_grid(k))


# Amelia, Michael, Minh
# Maps keypoint to cell
# returns a list of tuples
def keypointsToCells(img, keypoints):
    height = len(img)
    width = len(img[0])
    cells = []
    for point in keypoints:
        cells.append(( math.floor((point.pt[1]/width)*9), math.floor((point.pt[0]/height)*9)))
        
    return cells

# Amelia, Michael, Minh
# Prints out visual grid in terminal
# to see if keypoint mapping is correct
def testMapping(cells):
    grid = np.zeros((9,9))
    for c in cells:
        grid[c[0]][c[1]] = 1
    for row in grid:
        for index in row:
            if index == 0:
                print("\033[0;30;47mO", end = ", ")
            if index == 1:
                print("\033[0;31;47mX\033[0;30;47m", end = ", ")
        print("\033[39;49m")
    print()
    


'''
End of group submissions
'''



'''
BLOCK FOR TESTING STUFF
Please put a #comment above the group blocks quickly explaining what they're used for.
The green will stand out and make it easier to find stuff.
'''
#Runs through Sudoku boards in images directory, 
# prints the board and displays the original image

#Currently doesn't work due to resizing errors and None-type list errors.
#Seems to be keypoint related, perhaps the dilation is messing it up.
'''
#Quick loop for testing against several Sudoku boards
#num is how many boards to check
def test_sudoku_images(num: int):
    for i in range(num):
        path = "images/sudoku" + str(i) + ".png"
        img_orig = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        board = sudoku_image_to_board(img_orig)
        pretty_print_board(board)
        cv2.imshow(path, img_orig)
        while cv2.waitKey(1) & 0xFF != ord('n'):
            continue
'''

#Testing delete_grid method
img = cv2.imread("images/sudoku0.png", cv2.IMREAD_GRAYSCALE)
print("Original size: " + str(img.shape))
cv2.imshow("Hello there", img)
img2 = delete_grid(img, 100)
print("New size: " + str(img2.shape))
cv2.imshow("Did somebody say...less grid?", img2)
cv2.waitKey()

#Prints out the Sudoku board nicely 

#Pretty-printing for the Sudoku board
def pretty_print_board(board):
    for l in board:
        print(l)
'''
img_orig = cv2.imread("images/sudoku2.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Here's a picture...", img_orig)
while cv2.waitKey(1) & 0xFF is not ord('n'):
    continue
pretty_print_board(sudoku_image_to_board(img_orig))
'''

#Tests keypoints to cells methodj
'''
testMapping((keypointsToCells(img,keypoints)))
'''


#Displays circles according to where they should (hopefully) be in a grid
'''
#Now write a method that displays circles in the same spot as they are, using map_keypoints
def draw_keypoint_grid(list_of_keypoints: list, size: int):
    x_dictionary = map_keypoints(list_of_keypoints, 10, "x")
    y_dictionary = map_keypoints(list_of_keypoints, 10, "y")
    img = 255 * np.ones(shape=[size, size, 3], dtype=np.uint8)
    spacing = size//9
    buffer = 15
    for kp in list_of_keypoints:
        column = x_dictionary.get(get_x_position(kp))
        row = y_dictionary.get(get_y_position(kp))
        center = (column*spacing + buffer, row*spacing + buffer)
        img = cv2.circle(img, center, 15, (255, 0, 0), 2)
    cv2.imshow("Grid!", img)
    cv2.waitKey()
'''


#Shows keypoints, then the keypoint image snips
'''
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)

for k in keypoints:
    size = int(k.size)
    p = tuple(int(x-size/2) for x in k.pt)
    print(p)
    cv2.rectangle(img, p, (p[0]+size, p[1]+size), 200, 5)

    digit_img = img_orig[p[1]:p[1]+size, p[0]:p[0]+size]
    cv2.imshow("One Digit", digit_img)
    while cv2.waitKey(1) & 0xFF != ord('n'):
        continue
cv2.imshow("Blob Rectangles", img)
'''


#Video capture from webcam
'''
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#cap.release()

'''


'''
End of testing block
'''


'''
Clean up
'''
# When everything done, release the capture
cv2.waitKey()
cv2.destroyAllWindows()

