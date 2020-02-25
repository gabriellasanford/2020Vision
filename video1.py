import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

img_orig = cv2.imread("../sudoku.png", cv2.IMREAD_GRAYSCALE)

# Show the original image
# This is a matplotlib display, so we must close the window to move forward
plt.imshow(255-img_orig, cmap=plt.cm.binary)
plt.show()

element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7), (3, 3))
img = 255-cv2.dilate(255-img_orig, element)
# This is a matplotlib display, so we must close the window to move forward
plt.imshow(255-img, cmap=plt.cm.binary)
plt.show()

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Filter by Inertia
#params.filterByInertia = True
#params.minInertiaRatio = 0.2

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)
 
# Detect blobs.
keypoints = detector.detect(img)
print([(k.pt, k.size) for k in keypoints])


# deduce what sudoku cell each keypoint is in.
# returns a list of tuples (i, j) where is i is
# the row index and j is the column index.
def duy_paul_gabriella_keypoints_to_cells(keypoint_list):
    left = sys.maxsize
    right = 0
    top = sys.maxsize
    bottom = 0
    cells = []
    for a_keypoint in keypoint_list:
        left = min(left, a_keypoint.pt[0])
        right = max(right, a_keypoint.pt[0])
        top = min(top, a_keypoint.pt[1])
        bottom = max(bottom, a_keypoint.pt[1])
    # distance from one column center to the next
    x_bucket_size = (right - left) / 8
    # distance from one row center to the next
    y_bucket_size = (bottom - top) / 8
    for a_keypoint in keypoint_list:
        x, y = a_keypoint.pt
        j = round((x - left) / x_bucket_size)
        i = round((y - top) / y_bucket_size)
        cells.append((i, j))
    return cells


# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the
# size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(img, keypoints, None, (255,0,255),\
                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
###
### Here we start using imshow() --- the native OpenCV image viewer
### These windows do not lock up the main thread
###

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)


for k in keypoints:
    size = int(k.size)
    p = tuple(int(x-size/2) for x in k.pt)
    print(p)
    cv2.rectangle(img, p, (p[0]+size, p[1]+size), 200, 5)

    digit_img = img_orig[p[1]:p[1]+size, p[0]:p[0]+size]
    cv2.imshow("One Digit", digit_img)
    while cv2.waitKey(1) & 0xFF != ord('n'):
        continue
cv2.imshow("Blob Rectangles", img)



cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
