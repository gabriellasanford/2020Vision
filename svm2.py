import cv2
import numpy as np
import time

WIDTH = 900
HEIGHT = 900
BORDER = 10
image_size = 28
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), \
          (0, 255, 255), (100, 100, 255), (255, 100, 255), (100, 255, 100), (255, 100, 100)]

"""
    FEATURE FUNCTIONS
"""

#Will, Gabriella, Amelia
#hv_weights on canny-edged image
#Results: 79/57
def hv_weights_on_canny(img):
    converted_img = np.uint8(img)
    edges = cv2.Canny(converted_img,230,250)
    return hv_weights(edges)



# Number of b/w transitions along every other row
# 14 dimensions
# Rob Hochberg
def waviness(img):
    img2 = img.copy()
    img2[img2 > 0] = 255 # Any pixel not white becomes black
    return np.sum(abs(img2[:,1:] - img2[:,:-1])/255, axis=1)[::2].tolist()

# Waviness above, but performed after doing edge detection
def edginess(img):  
    img2 = np.uint8(img)
    edges = cv2.Canny(img2,100,200)
    #print (edges)
    return waviness(edges)

# Waviness above, but performed after doing edge detection
def Sobelness(img):  
    img2 = np.uint8(img)
    edges = cv2.cv2.Sobel(img2,cv2.CV_64F,1,0,ksize=5)
    #print (edges)
    return waviness(edges)


# For each row, count number of non-white pixels.
# For each column, count number of non-white pixels.
# Total dimension: 28 x 2 = 56.
# Duy, Michael Bujard, Paul
def hv_weights(image):
    # row_nonzero_counts = np.count_nonzero(image, axis=1)
    # col_nonzero_counts = np.count_nonzero(image, axis=0)
    row_nonzero_counts = np.asarray([sum([0 if num == 0 else 1 for num in a_row]) for a_row in image])
    col_nonzero_counts = np.asarray([sum([0 if num == 0 else 1 for num in a_col]) for a_col in np.transpose(image)])
    #print "Hi", np.concatenate((row_nonzero_counts, col_nonzero_counts))
    return np.concatenate((row_nonzero_counts, col_nonzero_counts)).tolist()


# Anthony/Amelia/Sri
# Count the vertical straight lines in an image,
# taking an image which is not necessarily blocked in black beforehand
vertical_line_len = 10
def vertical_lines(image):
    lines = [0] # Have a 0 to the left of the first char in the list                                                                  
    for x in range(image_size - 1):
        counter = 0
        max = 0
        for y in range(image_size - 1):
            pixel = 1 if image[y][x] > 0 else 0
            if counter == 0:
                counter += pixel
            else:
                if pixel == 1:
                    counter += pixel
                else: #if there's a gap                                                                                                               
                    max = np.maximum(max, counter)
                    counter = 0
        max = np.maximum(max, counter)
        lines.append(1 if max >= vertical_line_len else 0)
    #print(lines)                                                                                                                                     
    line_count = 0
    for i in range(1, image_size):
        if lines[i] == 0 and lines[i-1] == 1:
            line_count += 1
    return [line_count]


# Minh/Matt/David
# Build the vertical waviness of the image, and sum it with the horizontal waviness.
# gets horizontal waviness, rotates matrix 90 degrees and runs formula for horizontal 
# waviness again for vertical waviness, add both waviness features together 
def combineWavy(img):
    x = np.sum(abs(img[1:]-img[:-1])/255,axis=1)[::2]
    img = np.rot90(img)
    y = np.sum(abs(img[1:]-img[:-1])/255,axis=1)[::2]
    z = np.add(x,y)
    return z.tolist()


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
 

# Alex, Eniola, Yeabkal
# Divides image into 49, 4x4 cells.
# Calculates the percentages of the total pixels within the cells that make up the image.
# Returns a 49 dimensional vector.
def sectional_density(image):
    CELL_WIDTH, CELL_HEIGHT = 4, 4
    pixel_percentages = [0 for i in range((image_size // CELL_WIDTH) * (image_size // CELL_HEIGHT))]
    total_black_pixels, count = 0, 0

    for corner_y in range(0, (image_size - CELL_HEIGHT + 1), CELL_HEIGHT):
        for corner_x in range(0, (image_size - CELL_WIDTH + 1), CELL_WIDTH):
            for i in range(CELL_HEIGHT):
                for j in range(CELL_WIDTH):
                    if image[corner_y + i][corner_x + j] > 0: # Pixel is black.
                        pixel_percentages[count] += 1
                        total_black_pixels += 1
            count += 1
    # Convert to percentages.
    for i in range(len(pixel_percentages)):
        pixel_percentages[i] = 100.0*pixel_percentages[i]/total_black_pixels

    return pixel_percentages


# Slantness
# Convolves with 4 kernels: vertical, horizontal, NE and SE,
# We are interested in the ratios of their values
def slantiness(img):
    kernelNE = np.array([[-1, 0, 1], [0, 2, 0], [1, 0, -1]])/np.sqrt(8)
    kernelSE = np.array([[1, 0, -1], [0, 2, 0], [-1, 0, 1]])/np.sqrt(8)
    #kernelH = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])/np.sqrt(6)
    #kernelV = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])/np.sqrt(6)
    imNE = convolve(img, kernelNE)
    imSE = convolve(img, kernelSE)
    imNE[imNE > 255] = 255
    imNE[imNE < 0] = 0
    imSE[imSE > 255] = 255
    imSE[imSE < 0] = 0
    #print imSE
    return sectional_density(imNE) + sectional_density(imSE)


# David/Sri/Michael
# Returns the convex hull as a list of points.
def convex_hull(img: np.array) -> list:
    # Convert the image to an OpenCV-compatible format.
    compat_image = np.uint8(img)
    # Threshold the image.
    thresh_val, img2 = cv2.threshold(compat_image, 0, cv2.THRESH_OTSU,\
        cv2.THRESH_BINARY)
    # Find contours on the thresholded image.
    contour_points, contours = cv2.findContours(np.uint8(img2), cv2.RETR_TREE,\
        cv2.CHAIN_APPROX_SIMPLE)
    # Create a list to hold the convex hull points.
    return_points = list()
    hull = np.vstack(cv2.convexHull(np.float32(contour_points[0]), False))
    for arr in hull:
        for item in arr:
            return_points.append(item)
    # Return the convex hull list.
    return return_points

# Sobel Gradient
# The Sobel gradient (used properly on larger images than these digits) is a
# measure of how much the image changes at some location. Useful for edge detection.
# This implementation produces a list of largest gradient angles in each cell
def Sobel_gradient(img):
    #plt.imshow(img, cmap=plt.cm.binary)
    #plt.show()
    Sobelx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])/np.sqrt(1)
    Sobely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])/np.sqrt(1)
    imx = convolve(img, Sobelx)
    imy = convolve(img, Sobely)
    grad_mag = np.sqrt(np.square(imx) + np.square(imy))
    grad_angle = np.arctan2(imx, imy)
    #plt.imshow(grad_img, cmap=plt.cm.binary)
    #plt.show()

    CELL_WIDTH, CELL_HEIGHT = 4, 4
    grad_angles = [0 for i in range((image_size // CELL_WIDTH) * (image_size // CELL_HEIGHT))]
    index = 0

    for corner_y in range(0, (image_size - CELL_HEIGHT + 1), CELL_HEIGHT):
        for corner_x in range(0, (image_size - CELL_WIDTH + 1), CELL_WIDTH):
            mag_max = np.max(grad_mag[corner_y:corner_y+CELL_HEIGHT,corner_x:corner_x+CELL_WIDTH])
            for i in range(CELL_HEIGHT):
                for j in range(CELL_WIDTH):
                    if grad_mag[corner_y + i][corner_x + j] == mag_max: # found greatest gradient
                        grad_angles[index] = grad_angle[corner_y + i][corner_x + j]
            index += 1
    #print grad_angles
    return grad_angles


# Finds centers of circles in the image, with various radii
def Hough_circles(img):
    #plt.imshow(img, cmap=plt.cm.binary)
    #plt.show()
    rmin, rmax = 2, 2
    hough = np.zeros((rmax-rmin+1, image_size, image_size))
    for imy in range(image_size):
        for imx in range(image_size):
            for r in range(rmin, rmax+1, 1):
                theta = 1.0/r
                for i in range(int(2 * np.pi * r)):
                    angle = theta * i
                    x, y = int(imx + r * np.cos(angle)), int(imy + r * np.sin(angle))
                    if x >= image_size or y >= image_size or x < 0 or y < 0: continue
                    hough[r-rmin][imy][imx] += img[y][x]
    hough[0] = hough[0] / (np.max(hough[0])/100.0)
    #hough[0][hough[0] < 30] = 0
    #plt.imshow(hough[0], cmap=plt.cm.binary)
    #plt.show()
    return sectional_density(hough[0])





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


# im is the target image
# k is the kernel
# returns the convolution image, without reversing k
def convolve(im, k):
    kh, kw = k.shape
    imh, imw = im.shape
    im_w_border = np.zeros((kh + imh - 1, kw + imw -1))
    im_w_border[(kh-1)//2:(kh-1)//2+imh, (kw-1)//2:(kw-1)//2+imw] += im
    new_img = np.array([[np.sum(k*im_w_border[i:i+kh, j:j+kw]) \
                for j in range(imw)] for i in range(imh)], dtype='float')
    new_img[new_img>255] = 255
    new_img[new_img<0] = 0
    
    return new_img



"""
    TESTS
"""
# feature_map maps digits to list of list of features
# com_map maps digits to a com of that digit's features
def testR(labels, pred):
    predictions = [[0 for i in range(10)] for j in range(10)]
    for i in range(len(labels)):
        predictions[labels[i]][int(pred[i])] += 1
    return predictions


# feature_map maps digits to list of list of features
# com_map maps digits to a com of that digit's features
def testAMD(labels, pred):
    countr = 0.0
    countw = 0.0
    for i in range(len(labels)):
        if labels[i] == int(pred[i]):
            countr += 1
        else:
            countw += 1
    return countr/(countr +  countw)


"""
    TRAINING AND TESTING
"""
# Select the set of features
features = [waviness, slantiness]

# # Set up training data
# print("Extracting Features.", end=" ")
# start = time.time()
# data = read_images("data/mnist_train.csv")
# digit_map = make_digit_map(data)
# feature_map = build_feature_map(digit_map, features)
# train = []
# train_labels = []
# #print(feature_map)
# for digit in range(10):
#     #print(digit, len(feature_map[digit]))
#     for f in feature_map[digit]:
#         train.append(f)
#         train_labels.append(digit)
# print(int(time.time() - start), "seconds.")

# # Train the SVM
# print("Beginning training.", end=" ")
# start = time.time()
# svm = cv2.ml.SVM_create()
# #svm.setType(cv2.ml.SVM_C_SVC)
# #svm.setKernel(cv2.ml.SVM_LINEAR)
# svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
# #svm.train(np.array(train).astype(np.float32), cv2.ml.ROW_SAMPLE, np.array(train_labels))
# svm.trainAuto(np.array(train).astype(np.float32), cv2.ml.ROW_SAMPLE, np.array(train_labels))
# print(int(time.time() - start), "seconds.")

# Read SVM from file (comment out if training)
svm = cv2.ml_SVM.load("data/wav-slant-lrg-lrg.svm")

# Test the SVM
print("Beginning testing.", end=" ")
start = time.time()
data = read_images("data/mnist_test.csv")
digit_map = make_digit_map(data)
feature_map = build_feature_map(digit_map, features)
test = []
test_labels = []
for digit in range(10):
    for f in feature_map[digit]:
        test.append(f)
        test_labels.append(digit)
predictions = svm.predict(np.array(test).astype(np.float32))[1]

print(int(time.time() - start), "seconds.")
print( "R Test\n", np.array(testR(test_labels, predictions)))
print("AMD Test", testAMD(test_labels, predictions))
#for i in range(len(test)):
#    print(i, test_labels[i], predictions[i])


