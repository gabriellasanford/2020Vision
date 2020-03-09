import hough_grid as hagrid
import cv2
import math
import random as rnd
import numpy as np
from collections import Counter

# Constants
# Canny thresholds
THRESH_ONE = 10
THRESH_TWO = 40

# HoughLinesP parameters
RHO = 1
THETA = math.pi/180
LINE_THRESH = 40
MIN_LENGTH = 50.0
MAX_GAP = 5.0

# Line drawing thickness
THICKNESS = 3

# Line drawing colors
BLACK = (255, 0, 0)

# Digit image dimensions
DIGIT_WIDTH = 28
DIGIT_HEIGHT = 28
DIGIT_CHANNELS = 1

# Read image. White = 255, Black = 0
img_orig = 255-cv2.imread("sudoku_square/sudoku5.png", cv2.IMREAD_GRAYSCALE)
#Mrot = cv2.getRotationMatrix2D((img_orig.shape[0]/2, img_orig.shape[1]/2), \
#                               45, 1)
#img_orig = cv2.warpAffine(img_orig, Mrot, (img_orig.shape))
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3), (1, 1))
#img_orig = cv2.dilate(img_orig, element)

# Accepts a grayscale image and returns list of edges from Canny edge detector
def canny_edges(gray_img):
    return cv2.Canny(gray_img, THRESH_ONE, THRESH_TWO)

def randcol():
    return (rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255))

# Returns the square of the length of a segment (a, b)-(c, d)
# given as a list [a, b, c, d]
def length2(l):
    a, b, c, d = l
    return ((b-a)*(b-a) + (d-c)*(d-c))

# Find the intersection of:
# The line through (a1, b1) and (c1, d1), and
# The line through (a2, b2) and (c2, d2)
def line_intersection(lop):
    [a1, b1, c1, d1, a2, b2, c2, d2] = lop
    det = (d1-b1)*(a2-c2) - (a1-c1)*(d2-b2)
    det1 = (a1*d1 - b1*c1)
    det2 = (a2*d2 - b2*c2)
    x = (det1*(a2 - c2) - det2*(a1 - c1)) / det
    y = (det2*(d1 - b1) - det1*(d2 - b2)) / det
    return (x, y)

# Draws segments (a, b), (c, d) in a list like [[a, b, c, d], [a, b, c, d], ...]
# atop a copy of the image img, using color. If color is -1, use random colors
def add_lines(img, segs, color):
    result = img.copy()
    if len(img.shape) < 3:
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # color
    #hough = np.ones(edges.shape, dtype=np.uint8) # b/w
    for i in range(0, len(segs)):
        l = segs[i][0]
        c = randcol() if color == -1 else color
        cv2.line(result, (l[0], l[1]), (l[2], l[3]), c, 3, cv2.LINE_AA)
    return result


# Takes a list of segments as given by a Hough transform [a, b, c, d]
# and returns a list in the same order as lines that clusters the angles.
# Also returns a list of labels sorted by frequency
def principal_angles(lines):
    # Build a list of angles for all the Hough segments
    angles = [-np.arctan((l[0][3]-l[0][1])/(l[0][2]-l[0][0])) for l in lines]

    # Cluster them to find the principle angles
    num_centers = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, a_labels, a_centers = cv2.kmeans(np.asarray(angles, dtype='float32'), num_centers, \
                        None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Build a dictionary mapping each label (0, 1, 2, ...) to its frequency in the list
    a_labels_map = Counter(a_labels.ravel().tolist())
    print("angle map", a_labels_map)
    kv = [(a_labels_map[i], i) for i in a_labels_map]
    kv.sort()
    print("KV", kv)
    return (a_labels, [x[1] for x in kv])


# Given a list segs of segments all at roughly the same angle, fit them to an equally-space
# grid of K lines. Return indices of two of the segments that bound the grid
def segment_bounds(segs, K):
    # Make a list of radii for the segments having the most frequent label/angle
    # We use the fact that the equation of the line at angle theta and distance R from
    #  the origin has equation R = x cos(theta) + y sin(theta)    # Build a list of angles for all the Hough segments
    angles = [-np.arctan((l[0][3]-l[0][1])/(l[0][2]-l[0][0])) for l in segs]
    radii_idx = [(segs[i][0][0]*np.sin(angles[i]) + segs[i][0][1]*np.cos(angles[i]), i) \
                 for i in range(len(segs))]
    radii_idx.sort()
    print(radii_idx)
    #print(radii_idx)
    radii = [r[0] for r in radii_idx]

    # Now try to fit the segments to the best ten equally-spaced radii
    minbadness = float("inf")
    bestpair = (0, 0)
    for i in range(len(radii)):
        for j in range(i+K-1, len(radii)):
            step = (radii[j] - radii[i])/(K-1)
            badness = 0
            for k in range(len(radii)):
                seg = radii[k]
                thisbad = 0
                if seg <= radii[i]: thisbad = radii[i] - seg
                elif seg >= radii[j]: thisbad = seg - radii[j]
                else:
                    thisbad = (seg - radii[i]) % step
                    thisbad = min(thisbad, step-thisbad)
                    thisbad = thisbad * thisbad # square penalty for inside the grid
                #print(step, thisbad)
                badness += thisbad * length2(segs[radii_idx[k][1]][0])
            if badness < minbadness:
                minbadness = badness
                bestpair = (i, j)
    return (radii_idx[bestpair[0]][1], radii_idx[bestpair[1]][1])


# Finds the four segments bounding a sudoku board img
def sudoku_bounds(img):
    cv2.imshow("Orig", img)

    # Canny edge detection
    edges = canny_edges(img)
    cv2.imshow("Canny", edges)

    # Perform the line detection
    lines = cv2.HoughLinesP(edges, RHO, THETA, LINE_THRESH,\
        minLineLength=MIN_LENGTH, maxLineGap=MAX_GAP)
    cv2.imshow("Hough", add_lines(img, lines, -1))

    # Find the principle angles
    a_labels, freqs = principal_angles(lines)
    lines0 = [lines[i] for i in range(len(lines)) if a_labels[i] == freqs[2]]
    lines1 = [lines[i] for i in range(len(lines)) if a_labels[i] == freqs[1]]
    imga = add_lines(img, lines0, (255, 0, 0))
    imga = add_lines(imga, lines1, (0, 0, 255))
    cv2.imshow("Two Max Angles", imga)

    # Find bounding lines for angle0
    l0, l1 = [lines0[i] for i in segment_bounds(lines0, 10)]
    m0, m1 = [lines1[i] for i in segment_bounds(lines1, 10)]
    bounds_img = add_lines(img, [l0, l1, m0, m1], (255, 255, 0))

    # Find the four corners
    c0 = line_intersection(l0[0].ravel().tolist() + m0[0].ravel().tolist())
    c1 = line_intersection(l0[0].ravel().tolist() + m1[0].ravel().tolist())
    c2 = line_intersection(l1[0].ravel().tolist() + m1[0].ravel().tolist())
    c3 = line_intersection(l1[0].ravel().tolist() + m0[0].ravel().tolist())
    
    cv2.circle(bounds_img, tuple(map(int, c0)), 5, (255, 0, 255), 5)
    cv2.circle(bounds_img, tuple(map(int, c1)), 5, (255, 0, 255), 5)
    cv2.circle(bounds_img, tuple(map(int, c2)), 5, (255, 0, 255), 5)
    cv2.circle(bounds_img, tuple(map(int, c3)), 5, (255, 0, 255), 5)
    cv2.imshow("Bounds", bounds_img)


#Video capture from webcam
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sudoku_bounds(gray)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
cap.release()
cv2.destroyAllWindows()
