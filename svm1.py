import cv2 as cv
import numpy as np


WIDTH = 900
HEIGHT = 900
BORDER = 10
image_size = 28
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), \
          (0, 255, 255), (100, 100, 255), (255, 100, 255), (100, 255, 100), (255, 100, 100)]

###
### Features
###

# Michael, Michael, and Will
# Feature: top-heavy vs bottom-heavy 
# "bottom-heavy" means that there's more going on in the bottom of the picture/digit.  
#   For example: 6 is bottom-heavy because there are more transitions in the bottom of the digit.
# "top-heavy" means that there's more going on in the top of the picture/digit.  
#   For example: 4 and 9 are top-heavy as there are more transitions in the top of the digit
# Returns a list with the top-weight and bottom-weight as (top, bottom)
def top_bottom_balance(img):
    # Get the number of color transitions per row in the image
    transition_array = color_transition_array(img)
    midpoint = len(transition_array)//2 #Get the midpoint of the array
    # Split the transition array into top and bottom of the image
    top_array = transition_array[:midpoint]
    bottom_array = transition_array[midpoint:]
    # Sum the values for number of color transitions in the top and bottom of the picture
    top_value = np.sum(top_array)
    bottom_value = np.sum(bottom_array)
    return [top_value, bottom_value]

# Michael, Michael, and Will
# Split the image in half and compare the weights
# (# of color transitions, can be easily modified to do sum of non-white
# pixel values) of the two halves.
# The function returns a tuple in the form (top_half, bottom_half), 
# with the bigger number representing which part of the image has more going on
# Returns a single array with the number of color transistions per row, 
# corresponding to that index in the returned array
def color_transition_array(img): 
    img2 = img.copy()
    img2[img2 > 0] = 255 # Any pixel not white becomes black
    return (np.sum(abs(img2[:, 1:] - img2[:, :-1])/255, axis=1))




"""
    Admin Functions
"""
# Read from our file
def read_images(filename):
    data = np.loadtxt(filename, delimiter=",", dtype='int')
    return data


# Dictionary to map each digit to its list of images
def make_digit_map(data):
    digit_map = {i:[] for i in range(10)}
    for row in data:
        digit_map[row[0]].append(row[1:].reshape((image_size, image_size)))
    return digit_map


# Extract features
# fnlist is a list of feature-generating functions, each of which should
#   take a 28x28 grayscale (0-255) image, 0=white, and return a 1-d array
#   of numbers
# Returns a map: digit -> nparray of feature vectors, one row per image
def build_feature_map(digit_map, fnlist):
    fmap = {i:[] for i in range(10)}
    for digit in fmap:
        for img in digit_map[digit]:
            feature_vector = []
            for f in fnlist:
                feature_vector += f(img)
            fmap[digit].append(feature_vector)
    return fmap


# Set up training data
data = read_images("data/mnist_medium.csv")
digit_map = make_digit_map(data)
feature_map = build_feature_map(digit_map, [top_bottom_balance])
train = []
labels = []
print(feature_map)
for digit in range(10):
    print(digit, len(feature_map[digit]))
    for f in feature_map[digit]:
        train.append(f)
        labels.append(digit)

# Train the SVM
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_LINEAR)
#svm.setKernel(cv.ml.SVM_RBF)
svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
svm.train(np.array(train).astype(np.float32), cv.ml.ROW_SAMPLE, np.array(labels))


# Data for visual representation
maxx, maxy = np.max(train, axis=0)

image = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8)*255
# Show the training data
for i in range(len(train)):
    y, x = int(train[i][1] * (HEIGHT-BORDER) / maxy) - BORDER, int(train[i][0] * (WIDTH-BORDER) / maxx) - BORDER
    color = colors[labels[i]]
    cv.circle(image, (x, y), 5, color, -1)


for i in range(0, image.shape[0], 2):
    for j in range(0, image.shape[1], 2):
        tx = (j + BORDER)*maxx*1.0 / (WIDTH - BORDER)
        ty = (i + BORDER)*maxy*1.0 / (WIDTH - BORDER)
        sampleMat = np.matrix([[tx, ty]], dtype=np.float32)
        response = int(svm.predict(sampleMat)[1][0])
        image[i,j] = colors[response]

cv.imshow('SVM Simple Example', image) # show it to the user
cv.waitKey()




