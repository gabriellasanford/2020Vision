import cv2
import numpy as np


def record_image(image_name, image):
    # cv2.imwrite(image_name, image)
    cv2.imshow(image_name, image)
    cv2.waitKey()


def drawline(line, frame):
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 5)


img = cv2.imread("images/sudoku31.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (800, 800))
# detect lines
record_image('original.png', img)
blank = np.zeros(img.shape, np.uint8)
edges = cv2.Canny(img, 50, 150, apertureSize=5)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
if lines is not None:
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    for line in lines:
        drawline(line, blank)
record_image('lines.png', blank)
# detect squares
size = 81
kernel = np.negative(np.ones((size, size)))
positive_value = 5
for j in range(size):
    kernel[0][j] = positive_value
    kernel[-1][j] = positive_value
for i in range(size):
    kernel[i][0] = positive_value
    kernel[i][-1] = positive_value
kernel = kernel / np.linalg.norm(kernel)
blob_image = cv2.filter2D(blank, 0, kernel)
record_image('convolved.png', blob_image)
# find blobs
params = cv2.SimpleBlobDetector_Params()
params.filterByConvexity = False
params.minDistBetweenBlobs = 20
params.minArea = 1
params.filterByColor = False
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(blob_image)
print(len([(k.pt, k.size) for k in keypoints]))
# draw detected blobs as red circles
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
image_with_keypoints = cv2.drawKeypoints(img, keypoints, None, (255, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
record_image('keypoints.png', image_with_keypoints)
