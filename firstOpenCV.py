import numpy as np
import cv2
from matplotlib import pyplot as plt
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
    print(feature_map[0])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    print(np.asarray(feature_map[0], dtype='float32'))
    ret,label,center=cv2.kmeans(np.asarray(feature_map[0], dtype='float32'), 2, None, criteria, 10,cv2.KMEANS_RANDOM_CENTERS)
    print(center)
    return com_map


features = [meth.waviness]
data = read_images("data/mnist_medium.csv")
digit_map = make_digit_map(data)
feature_map = build_feature_map(digit_map, features)
com = find_com(feature_map)

for d in range(10):
    for item in range(2):
        img = digit_map[d][item]
        img2 = np.uint8(img) # convert to a format for OpenCV
        #edges = cv2.Canny(img2,100,200)
        edges = cv2.Sobel(img2,cv2.CV_64F,1,1,ksize=5)
        #print(edges)
        plt.subplot(121)
        plt.imshow(img2, cmap=plt.cm.binary)
        plt.subplot(122)
        plt.imshow(edges,cmap=plt.cm.binary)
        plt.show()
