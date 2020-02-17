import numpy as np
import cv2
from matplotlib import pyplot as plt

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



data = read_images("data/mnist_medium.csv")
digit_map = make_digit_map(data)

for d in range(10):
    for item in range(2):
        img = digit_map[d][item]
        img2 = np.uint8(img)
        #edges = cv2.Canny(img2,100,200)
        edges = cv2.Sobel(img2,cv2.CV_64F,1,0,ksize=5)
        print(edges)
        plt.subplot(121)
        plt.imshow(img2, cmap=plt.cm.binary)
        plt.subplot(122)
        plt.imshow(edges,cmap=plt.cm.binary)
        plt.show()
