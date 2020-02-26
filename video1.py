import numpy as np
import cv2
from matplotlib import pyplot as plt
import knnClassify as knn

# stores a trained knnClassify. Initiated as None to avoid
# unnecessary overhead if not used
classifier = None


def make_classifier(feature):
    training_map = {}

    fonts = [cv2.FONT_HERSHEY_SIMPLEX,\
             cv2.FONT_HERSHEY_DUPLEX,cv2.FONT_HERSHEY_COMPLEX,\
             cv2.FONT_HERSHEY_TRIPLEX,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,\
             cv2.FONT_HERSHEY_SCRIPT_COMPLEX,cv2.FONT_ITALIC]
    
    for i in range(0,10):
        training_map[i] = []
        for k in range(0,50):
            img = np.zeros((28, 28,3), dtype=np.uint8)
            img = cv2.putText(img, str(i), (5+k%2,22+k%3), fonts[k%len(fonts)],\
                              0.85, (255,255,255), 2, cv2.LINE_AA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #img = cv2.bitwise_not(img)
            training_map[i].append(img)
            """cv2.imshow("digit"+str(i), img)
            while cv2.waitKey(1) & 0xFF != ord('n'):
                continue"""

    classifier = knn.make_trained_knn(feature, training_map)

    knn.test_existing_knn(knn.slantiness, classifier, training_map)


    return classifier

# Sri, Anthony, Eniola
# takes an np.array of digit images and returns an np.array of their respective digits
def classify_imgs(digit_imgs):
    feature = knn.slantiness
    
    digit_imgs = np.array(list(map(lambda i: cv2.resize(i, (28, 28)), digit_imgs)))
    global classifier
    if classifier is None:
        classifier = make_classifier(feature)

    """correct_classifications = [8,8,8,8,8,9,9,9,9,6,6,6,6,6,7,5,1,4,2,2,7,1,3,4,3,\
                               5,1,7,3,5]
    correct_map = {i:[] for i in range(10)}
    correct_count = {i:0 for i in range(10)}
    for i in range(len(correct_classifications)):
        correct_map[correct_classifications[i]].append(digit_imgs[i])
        correct_count[correct_classifications[i]] += 1
    print(correct_count)
    knn.test_existing_knn(feature, classifier, correct_map)"""

    result = []
    for digit in digit_imgs:
        img = cv2.resize(digit, (28, 28))
        img = cv2.bitwise_not(img)
        predict = knn.classify_digit(classifier, np.array(feature(img)))
        result.append(predict)
        print(predict)
        #print(img)
        cv2.imshow("One Digit", img)
        while cv2.waitKey(1) & 0xFF != ord('n'):
            continue
    
    return np.array(result)
    

img_orig = cv2.imread("sudoku.png", cv2.IMREAD_GRAYSCALE)

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

#stores all found digits (remove for push)
all_digit_imgs = []

for k in keypoints:
    size = int(k.size)
    p = tuple(int(x-size/2) for x in k.pt)
    print(p)
    cv2.rectangle(img, p, (p[0]+size, p[1]+size), 200, 5)

    digit_img = img_orig[p[1]:p[1]+size, p[0]:p[0]+size]
    all_digit_imgs.append(digit_img) #(remove for push)
    """cv2.imshow("One Digit", digit_img)
    while cv2.waitKey(1) & 0xFF != ord('n'):
        continue"""
print(classify_imgs(np.array(all_digit_imgs)))
cv2.imshow("Blob Rectangles", img)
"""
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
"""
