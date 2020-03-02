"""
* Read images
* Extract features from each image by convolving with filters
"""

import numpy as np
import matplotlib.pyplot as plt
import math
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
    
kernel = np.array([[-1, 1],[-1, 1]]) # left-edge detector
kernel = np.array([[-1, -1],[3, -1]])/3.0 # upper-right corner detector
#kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9.0
kernel1 = np.array([[-1, 0, 1], [0, 2, 0], [1, 0, -1]])/6.0
kernel2 = np.array([[1, 0, -1], [0, 2, 0], [-1, 0, 1]])/6.0
kernel3 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])/math.sqrt(6)
Sobelx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])/math.sqrt(12)
Sobely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])/math.sqrt(12)

data = read_images("data/mnist_medium.csv")
digit_map = make_digit_map(data)

imgr = digit_map[8][7].reshape((image_size, image_size))
plt.imshow(imgr, cmap=plt.cm.binary)
plt.show()

imgrc = meth.convolve(imgr, Sobelx)
plt.imshow(imgrc, cmap=plt.cm.binary)
plt.show()

imgrc = meth.convolve(imgr, Sobely)
plt.imshow(imgrc, cmap=plt.cm.binary)
plt.show()

imgrc = meth.convolve(imgr, kernel1)
plt.imshow(imgrc, cmap=plt.cm.binary)
plt.show()

imgrc = meth.convolve(imgr, kernel2)
plt.imshow(imgrc, cmap=plt.cm.binary)
plt.show()

imgrc = meth.convolve(imgr, kernel3)
plt.imshow(imgrc, cmap=plt.cm.binary)
plt.show()


