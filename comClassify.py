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
import methods as meth

image_size = 28 # width and length

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
def find_com(feature_map):
    com_map = {}
    for digit in feature_map:
        feature_matrix = np.asfarray(feature_map[digit])
        com_map[digit] = np.mean(feature_matrix, axis=0)
    return com_map
    

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
                dist = abs(np.linalg.norm(com_map[ckey]-ele))
                if dist <= mdist:
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
            for candidate_digit, a_com in com_map.items():
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
            for candidate_digit, a_com in com_map.items():
                distance = np.linalg.norm(a_com - some_features)
                if distance < smallest_distance:
                    smallest_distance = distance
                    guess_digit = candidate_digit
            predictions[correct_digit][guess_digit] += 1
    return predictions

# List of implemented feature functions
all_features = [meth.draw_and_quarter, meth.waviness, meth.hv_weights,\
                meth.top_bottom_balance, meth.combineWavy,\
                meth.vertical_lines, meth.sectional_density, meth.slantiness,\
                meth.edginess, meth.Sobelness]
all_features = [meth.convex_hull]

data = read_images("data/mnist_medium.csv")
digit_map = make_digit_map(data)

for f in all_features:
    print ("\n\n", f)

    features = [f]
    
    # train
    feature_map = build_feature_map(digit_map, features)
    com = find_com(feature_map)

    # Test on training data
    print("AMD Test", testAMD(feature_map, com))
    print("SPM Test", testSPM(feature_map, com))
    print( "R Test\n", np.array(testR(feature_map, com)))

    # Test on test data
    data = read_images("data/mnist_medium_test.csv")
    digit_map = make_digit_map(data)
    feature_map = build_feature_map(digit_map, features)
    print("AMD Test", testAMD(feature_map, com))
    print("SPM Test", testSPM(feature_map, com))
    print( "R Test\n", np.array(testR(feature_map, com)))

           

"""
data = read_images("mnist_medium.csv")
digit_map = make_digit_map(data)

for img in digit_map[8]:
    imgr = img.reshape((image_size*1, image_size/1))
    plt.imshow(imgr, cmap=plt.cm.binary)
    plt.show()
"""



           
