"""
* Read images
* Extract features from each image
* Find center of mass of features by label
* Classify images by nearest center of mass
* Test accuracy of classification on training data
* Test accuracy on test data
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

image_size = 28 # width and length

"""
    FEATURE FUNCTIONS
"""

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
    edges = cv2.Sobel(img2,cv2.CV_64F,1,0,ksize=5)
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
                    angle = theta * i;
                    x, y = int(imx + r * np.cos(angle)), int(imy + r * np.sin(angle))
                    if x >= image_size or y >= image_size or x < 0 or y < 0: continue
                    hough[r-rmin][imy][imx] += img[y][x]
    hough[0] = hough[0] / (np.max(hough[0])/100.0)
    #hough[0][hough[0] < 30] = 0
    #plt.imshow(hough[0], cmap=plt.cm.binary)
    #plt.show()
    return sectional_density(hough[0])



###############################################################################
# Quarter slicer
# A lot like sectional_density, but it works on variable size images, so it can
# be used in conjunction with whitespace trimming.
###############################################################################


# Tweakable constant that puts an upper limit on the number of times the image
# can be quartered.  On the first iteration, the image is cut into quarters, on
# the second, those quarters are cut into quarters, and so on.
MAX_RECURSIONS = 3


# Entry point function for compatibility with comClassify testing function.
def draw_and_quarter(img: np.array) -> list:
    trimmed_img = trim_whitespace(img)
    return quarter_density(trimmed_img, 0)


# Carves image of any size into quarters and returns a list with the weight
# (number of greyscale pixels) in each quarter.  Grabs weights recursively
# and puts them all in a list.
def quarter_density(image: np.array, iteration: int) -> list:
    iteration += 1
    # Get quarter slices, then weigh each slice
    quarters = get_quarter_slices(image)
    quarter_weights = list()
    if iteration is MAX_RECURSIONS:
        for quarter in quarters:
            quarter_weights += get_image_weight(quarter)
    else:
        for quarter in quarters:
            quarter_weights += quarter_density(quarter, iteration)
    return quarter_weights


# Slices the image into quarters and returns a list of quarters.
def get_quarter_slices(image: np.array) -> list:
    # Get the dimensions of the image, then cut them in half (to the nearest
    # integer).
    height, width = image.shape
    half_height = height // 2
    half_width = width // 2

    # Slice the image into quarters and store the quarters in a list.
    # List order is top-left, top-right, bottom-left, bottom-right.
    quarter_slices = list()
    quarter_slices.append(image[:half_height, :half_width])
    quarter_slices.append(image[:half_height, half_width:])
    quarter_slices.append(image[half_height:, :half_width])
    quarter_slices.append(image[half_height:, half_width:])

    return quarter_slices


# Accepts an image of any size and returns its weight, determined by the number
# of non-white pixels in the image.
def get_image_weight(image: np.array) -> list:
    weight = 0
    height, width = image.shape
    for row in range(height):
        for col in range(width):
            weight += image[row][col]
    return [weight]


#Method for picking out endpoints in images of digits
def endpoints(image: np.array) -> np.array:
    #Make the kernel for endpoint detection
    endpoint_kernel = np.array([[-2, -2, -2], [-2, 10, -2], [-2, -2, -2]])
    endpoint_kernel = endpoint_kernel/np.linalg.norm(endpoint_kernel)
    #Trim whitespace on the image
    r_img = trim_whitespace(image)
    #Threshold the image to turn it into a blocky, black digit
    #r_img = threshold_image(r_img, (5, 255), (6, 0))
    #Convolve the image to get rid of most non-endpoints
    r_img = convolve(r_img, endpoint_kernel)
    #Threshold again to clean up bad gray pixels
    #r_img = threshold_image(r_img, (50, 255), (52, 0))
    return r_img



###############################################################
# Function (and helpers) to trim whitespace from a digit image.
###############################################################


# Trims whitespace, leaving the digit centered in the image.
# Resizes trimmed image to 28x28.
# image -> image
def trim_whitespace(image):
    # (edge-near-top, edge-near-bottom, edge-near-left, edge-near-right)
    edge_positions = list() # List, will eventually have four items
    edge_positions.append(get_top_distance(image))
    edge_positions.append(get_bottom_distance(image))
    edge_positions.append(get_left_distance(image))
    edge_positions.append(get_right_distance(image))

    # Take out the piece containing the digit.
    # TODO: Slice out part of image containing digit.
    new_image = image[edge_positions[0]:(-edge_positions[1])+1, edge_positions[2]:(-edge_positions[3])+1]
    final_image = cv2.resize(new_image, (28, 28))
    return final_image


# Returns the number of the row in which the topmost non-white pixel is found.
def get_top_distance(image) -> int:
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if image[row][col] > 0:
                return row


# Returns the number of the row immediately after that in which the bottommost 
# non-white pixel is found.
def get_bottom_distance(image) -> int:
    for row in range(1, image.shape[0] + 1):
        for col in range(image.shape[1]):
            if image[image.shape[0] - row][col] > 0:
                return row 


# Returns the number of the column in which the leftmost non-white pixel is found.
def get_left_distance(image) -> int:
    for col in range(image.shape[1]):
        for row in range(image.shape[0]):
            if image[row][col] > 0:
                return col


# Returns the number of the column immediately to the right of that in which the
# rightmost non-white pixel is found.
def get_right_distance(image) -> int:
    for col in range(1, image.shape[1] + 1):
        for row in range(image.shape[0]):
            if image[row][image.shape[1] - col] > 0:
                return col



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


# Find center of mass of each digit's feature vectors
# feature_map is a map from each digit to a list of feature vectors
# Returns a map: digit -> Center of mass
def find_coms(feature_map):
    num_centers = 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    com_map = {}
    for digit in feature_map:
        feature_matrix = np.asfarray(feature_map[digit])
        #com_map[digit] = np.mean(feature_matrix, axis=0)
        ret,label,centers = cv2.kmeans(np.asarray(feature_map[digit], dtype='float32'), num_centers, \
                    None, criteria, 10,cv2.KMEANS_RANDOM_CENTERS)
        com_map[digit] = centers
    return com_map


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
# Anthony/Michael A./David
# feature_map maps digits to list of list of features
# com_map maps digits to a com of that digit's features
# Returns the fraction of correctly-classified images
def testAMD(feature_map, com_map):
    ccom = None
    countr = 0.0
    countw = 0.0
    for key in feature_map:
        for ele in feature_map[key]:
            mdist = float("inf")
            for ckey in com_map:
                for com in com_map[ckey]:
                    dist = abs(np.linalg.norm(com-ele))
                    if dist < mdist:
                        mdist = dist
                        ccom = ckey
            if ccom == key:
                countr +=1  
            else:
                countw +=1
    per = countr/(countr +  countw)
    return per 


# Sri / Paul / Michael Bujard
# Test
# feature_map maps digits to list of list of features
# com_map maps digits to a com of that digit's features
def testSPM(feature_map, com_map):
    num_correct = 0.0
    num_total = 0.0
    for correct_digit, list_of_features in feature_map.items():
        for some_features in list_of_features:
            smallest_distance = float("inf")
            guess_digit = None
            for candidate_digit, com_list in com_map.items():
                for a_com in com_list:
                    distance = np.linalg.norm(a_com - some_features)
                    if distance < smallest_distance:
                        smallest_distance = distance
                        guess_digit = candidate_digit
            if guess_digit == correct_digit:
                num_correct += 1
            num_total += 1
    return num_correct / num_total


# Rob Hochberg
# feature_map maps digits to list of list of features
# com_map maps digits to a com of that digit's features
def testR(feature_map, com_map):
    predictions = [[0 for i in range(10)] for j in range(10)]
    for correct_digit, list_of_features in feature_map.items():
        for some_features in list_of_features:
            smallest_distance = float("inf")
            guess_digit = None
            for candidate_digit, com_list in com_map.items():
                for a_com in com_list:
                    distance = np.linalg.norm(a_com - some_features)
                    if distance < smallest_distance:
                        smallest_distance = distance
                        guess_digit = candidate_digit
            predictions[correct_digit][guess_digit] += 1
    return predictions

# List of implemented feature functions
all_features = [draw_and_quarter, waviness, hv_weights, top_bottom_balance,\
                combineWavy, vertical_lines, sectional_density, slantiness,\
                edginess, Sobelness]
#all_features = []

for f in all_features:
    print ("\n\n", f)

    features = [f]
    
    # train
    data = read_images("data/mnist_medium.csv")
    digit_map = make_digit_map(data)
    feature_map = build_feature_map(digit_map, features)
    com = find_coms(feature_map)

    # Test on training data
    print("AMD Test", testAMD(feature_map, com))
    print("SPM Test", testSPM(feature_map, com))
    #print( "R Test\n", np.array(testR(feature_map, com)))

    # Test on test data
    data = read_images("data/mnist_medium_test.csv")
    digit_map = make_digit_map(data)
    feature_map = build_feature_map(digit_map, features)
    print("AMD Test", testAMD(feature_map, com))
    print("SPM Test", testSPM(feature_map, com))
    #print( "R Test\n", np.array(testR(feature_map, com)))

           

"""
data = read_images("mnist_medium.csv")
digit_map = make_digit_map(data)

for img in digit_map[8]:
    imgr = img.reshape((image_size*1, image_size/1))
    plt.imshow(imgr, cmap=plt.cm.binary)
    plt.show()
"""



           
