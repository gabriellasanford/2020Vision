import numpy as np
import matplotlib.pyplot as plt

image_size = 28 # width and length

# Read from our file
data = np.loadtxt("mnist_small.csv", delimiter=",") 

# Take a look, make sure they are floats
# print data


# Extract all but first column
imgs = np.asfarray(data[:, 1:])

# Grab first image
img = imgs[12].reshape((image_size*1, image_size/1))

# Draw it using matplotlib
plt.imshow(img, cmap=plt.cm.binary)
plt.show()

# threshold all gray to black
# copy the image
img2 = img.copy()
img2[img2 > 0] = 255
plt.imshow(img2, cmap=plt.cm.Greys)
plt.show()

def waviness(img):
    return np.sum(abs(img2[:,1:] - img2[:,:-1])/255, axis=1)[::2]
