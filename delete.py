import hough_grid as hagrid
import cv2
import math
import random as rnd
import numpy as np

# Constants
# Canny thresholds
THRESH_ONE = 10
THRESH_TWO = 200

# HoughLinesP parameters
RHO = 1
THETA = math.pi/180
LINE_THRESH = 50
MIN_LENGTH = 25
MAX_GAP = 5

# Line drawing thickness
THICKNESS = 3

# Line drawing colors
BLACK = (255, 0, 0)

# Digit image dimensions
DIGIT_WIDTH = 28
DIGIT_HEIGHT = 28
DIGIT_CHANNELS = 1

# Read image. White = 255, Black = 0
img_orig = 255-cv2.imread("images/sudoku20.png", cv2.IMREAD_GRAYSCALE)
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3), (1, 1))
img_orig = cv2.dilate(img_orig, element)

# Accepts a grayscale image and returns list of edges from Canny edge detector
def canny_edges(gray_img):
    return cv2.Canny(gray_img, THRESH_ONE, THRESH_TWO)

def randcol():
    return (rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255))

cv2.imshow("Orig", img_orig)
#cv2.waitKey()

edges = canny_edges(img_orig)
cv2.imshow("Canny", edges)

lines = cv2.HoughLinesP(edges, RHO, THETA, LINE_THRESH, MIN_LENGTH, MAX_GAP)
#print(lines)
#hough = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
hough = cv2.cvtColor(np.zeros(edges.shape, dtype=np.uint8)*255, cv2.COLOR_GRAY2BGR) # color
#hough = np.ones(edges.shape, dtype=np.uint8) # b/w
for i in range(0, len(lines)):
    l = lines[i][0]
    cv2.line(hough, (l[0], l[1]), (l[2], l[3]), randcol(), 3, cv2.LINE_AA)
cv2.imshow("Hough Lines", hough)


# Draw a white grid on a black background.
# Image size is HxW, grid dims = (rows, cols), width of grid lines = width
def make_grid(H, W, rows, cols, width):
    rv_grid = np.zeros((H, W), dtype=np.uint8)
    cellh = (H-width)//rows
    cellw = (W-width)//cols
    for i in range(cols+1):
        cv2.line(rv_grid, (width//2 + i*(W-width)//cols, 0), (width//2 + i*(W-width)//cols, H-1), 255, 3, cv2.LINE_AA)
    for i in range(rows+1):
        cv2.line(rv_grid, (0, width//2 + i*(H-width)//rows), (W-1, width//2 + i*(H-width)//rows), 255, 3, cv2.LINE_AA)
    return rv_grid



# Given an image that contains a white (255) grid on a black (0) background,
# And the number of rows/cols to look for, find best (A, B, C, D) 4-tuple
# of corner points
def find_best_grid(img, rows, cols):
    # Make a reference grid
    H, W = img.shape
    width_factor = 0.2 # fraction of image that is grid
    width = int(min(W*width_factor/(cols+1), H*width_factor/(rows+1)))
    grid_img = make_grid(H, W, rows, cols, width//2)
    cv2.imshow("Reference Grid", grid_img)
    
    # Now search for best warping
    step = int(1.5*width)
    B = 15 # 1/B is the region near corners where warp corners should be
    corners = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    high_score = 0
    print(H, W, width)
    for Ax in range(0, W//B, step):
        for Ay in range(0, H//B, step):
            for Bx in range((B-1)*W//B, W, step):
                for By in range(0, H//B, step):
                    for Cx in range((B-1)*W//B, W, step):
                        for Cy in range((B-1)*H//B, H, step):
                            for Dx in range(0, W//B, step):
                                for Dy in range((B-1)*H//B, H, step):
                                    warp_corners = np.float32([[Ax, Ay], [Bx, By], [Cx, Cy], [Dx, Dy]])
                                    warp_matrix = cv2.getPerspectiveTransform(corners, warp_corners)
                                    warp_image = cv2.warpPerspective(grid_img, warp_matrix, (W, H))
                                    score = sum(sum(np.multiply(warp_image, img)))
                                    if score > high_score:
                                        high_score = score
                                        print(score)
                                        cv2.imshow("Warped Grid", warp_image)
                                        #cv2.waitKey()
                
find_best_grid(img_orig, 9, 9)

print("We are done finding the best")
cv2.waitKey()
cv2.destroyAllWindows()
