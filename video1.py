import numpy as np
import cv2
from matplotlib import pyplot as plt
import knnClassify as knn

# stores a trained knnClassify. Initiated as None to avoid
# unnecessary overhead if not used
classifier = None


def make_classifier(feature):
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

    knn.test_existing_knn(knn.slantiness, classifier, training_map)


    return classifier

# Sri, Anthony, Eniola
# takes an np.array of digit images and returns an np.array of their respective digits
def classify_imgs(digit_imgs):
    feature = knn.slantiness
    
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
    

img_orig = cv2.imread("sudoku.png", cv2.IMREAD_GRAYSCALE)

# Show the original image
# This is a matplotlib display, so we must close the window to move forward
plt.imshow(255-img_orig, cmap=plt.cm.binary)
plt.show()

element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7), (3, 3))
img = 255-cv2.dilate(255-img_orig, element)
# This is a matplotlib display, so we must close the window to move forward
plt.imshow(255-img, cmap=plt.cm.binary)
plt.show()

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Filter by Inertia
#params.filterByInertia = True
#params.minInertiaRatio = 0.2

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)
 
# Detect blobs.
keypoints = detector.detect(img)
print([(k.pt, k.size) for k in keypoints])


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
METHODS FOR HOMEWORK
'''


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
    if axis is "x":
        list_of_points.sort(key=get_x_position)
    elif axis is "y":
        list_of_points.sort(key=get_y_position)
    #Make dictionaries to hold row and column mappings
    dictionary = dict()
    #Non-sense starter values 
    last_val = -5 * spread
    row_or_column = -1
    for k in list_of_points:
        position = get_position(k)
        if axis is "x":
            current_val = position[0]
        elif axis is "y":
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
NOT HOMEWORK ANYMORE
'''


'''
New homework!
'''
#Method for making a 2-d list for a Sudoku board
#Takes keypoints, classifies the digits puts them in the appropriate spot 
def keypoints_to_board(list_of_points: list):
    #Make row and column dictionaries 
    row_dict = map_keypoints(list_of_points, 10, "y")
    column_dict = map_keypoints(list_of_points, 10, "x")
    #Construct the Sudoku board as 2-d list, fill it with -1s as empty values
    board = [[-1 for col in range(9)] for row in range(9)]

    #Loop through and get the images to classify
    for k in keypoints:
        #Code courtesy of Dr. Hochberg for snipping the blobs
        size = int(k.size)
        p = tuple(int(x-size/2) for x in k.pt)
        cv2.rectangle(img, p, (p[0]+size, p[1]+size), 200, 5)
        digit_img = img_orig[p[1]:p[1]+size, p[0]:p[0]+size]
        #Now resize that to 28x28 for Anthony's method to get the value of the digit
        desired_size = (28, 28)
        digit_img = cv2.resize(digit_img, desired_size)
        '''Waiting on the method for this right here'''
        digit_val = 3
        #Get the position of the digit, then convert that to it's row and column position
        x = get_x_position(k)
        y = get_y_position(k)
        column = column_dict.get(x)
        row = row_dict.get(y)
        #Add the value of the digit to the Sudoku board
        board[row][column] = digit_val

    return board


#Pretty-printing for the Sudoku board
for l in keypoints_to_board(keypoints):
    print(l)


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

'''
# When everything done, release the capture
#cap.release()
cv2.destroyAllWindows()

