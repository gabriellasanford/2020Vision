"""
* Draw the pixels of circles of various sizes
"""

import numpy as np
import matplotlib.pyplot as plt

image_size = 28 # width and length


# Finds centers of circles in the image, with various radii
def drawCircle():
    r=12
    hough = np.zeros((image_size, image_size))
    theta = 1/r
    imx, imy = 14, 14
    oldx, oldy = imx, imy
    for i in range(int(2 * np.pi / theta)):
        angle = theta * i + 0.123;
        x, y = int(np.round(imx + r * np.cos(angle))),\
               int(np.round(imy + r * np.sin(angle)))
        if x >= image_size or y >= image_size or x < 0 or y < 0: continue
        if (x, y) == (oldx, oldy): continue
        oldx, oldy = x, y
        hough[y][x] += 100
    plt.imshow(hough, cmap=plt.cm.binary)
    plt.show()


drawCircle()


