"""
* Read images
* Extract features from each image by convolving with filters
"""

import numpy as np
import matplotlib.pyplot as plt
import math

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


# im is the target image
# k is the kernel
# returns the convolution image, without reversing k
def convolve(im, k):
    kh, kw = k.shape
    imh, imw = im.shape
    print im.shape
    print k.shape
    im_w_border = np.zeros((kh + imh - 1, kw + imw -1))
    im_w_border[(kh-1)/2:(kh-1)/2+imh, (kw-1)/2:(kw-1)/2+imw] += im
    new_img = np.array([[np.sum(k*im_w_border[i:i+kh, j:j+kw]) \
                for j in range(imw)] for i in range(imh)], dtype='int')
    print(new_img)
    new_img[new_img>255] = 255
    new_img[new_img<0] = 0
    
    return new_img
    




# im is the target image
# k is the kernel
# returns the convolution image, without reversing k
def convolveMax(im, k):
    return 0
    
kernel = np.array([[-1, 1],[-1, 1]]) # left-edge detector
kernel = np.array([[-1, -1],[3, -1]])/3.0 # upper-right corner detector
#kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9.0
kernel1 = np.array([[-1, 0, 1], [0, 2, 0], [1, 0, -1]])/6.0
kernel2 = np.array([[1, 0, -1], [0, 2, 0], [-1, 0, 1]])/6.0
kernel3 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])/math.sqrt(6)

data = read_images("mnist_medium.csv")
digit_map = make_digit_map(data)

imgr = digit_map[0][7].reshape((image_size*1, image_size/1))
plt.imshow(imgr, cmap=plt.cm.binary)
plt.show()

imgrc = convolve(imgr, kernel1)
plt.imshow(imgrc, cmap=plt.cm.binary)
plt.show()

imgrc = convolve(imgr, kernel2)
plt.imshow(imgrc, cmap=plt.cm.binary)
plt.show()

imgrc = convolve(imgr, kernel3)
plt.imshow(imgrc, cmap=plt.cm.binary)
plt.show()


