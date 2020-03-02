import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
import methods as meth

WIDTH = 900
HEIGHT = 900
BORDER = 10
image_size = 28
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), \
          (0, 255, 255), (100, 100, 255), (255, 100, 255), (100, 255, 100), (255, 100, 100)]


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


# Extract features
# digit_map maps each digit to a list of images
# fnlist is a list of feature-generating functions, each of which should
#   take a 28x28 grayscale (0-255) image, 0=white, and return a 1-d array
#   of numbers
# Returns a map: digit -> nparray of feature vectors, one row per image
def build_feature_map(digit_map, fnlist):
    fmap = {i:[] for i in range(10)}
    for digit in fmap:
        for img in digit_map[digit]:
            feature_vector = []
            for f in fnlist:
                feature_vector += f(img)
            fmap[digit].append(feature_vector)
    return fmap



"""
    TESTS
"""
# labels are the correct labels, pred gives the predicted labels.
def testR(labels, pred):
    predictions = [[0 for i in range(10)] for j in range(10)]
    for i in range(len(labels)):
        predictions[labels[i]][int(pred[i])] += 1
    return predictions


# labels are the correct labels, pred gives the predicted labels.
def testAMD(labels, pred):
    countr = 0.0
    countw = 0.0
    for i in range(len(labels)):
        if labels[i] == int(pred[i]):
            countr += 1
        else:
            countw += 1
            #plt.imshow(255-test_images[i], cmap=plt.cm.binary)
            #plt.show()
            print(labels[i], pred[i])
            print(test_images[i])
            
            #cv2.imshow("Incorrect", np.array(test_images[i], dtype=np.uint8))
            #cv2.waitKey()
    return countr/(countr +  countw)


"""
    TRAINING AND TESTING
"""
# Select the set of features
features = [meth.waviness, meth.slantiness]


# Set up training data
print("Extracting Features.", end=" ")
start = time.time()
data = read_images("data/mnist_medium_train.csv")
digit_map = make_digit_map(data)
feature_map = build_feature_map(digit_map, features)
train = []
train_labels = []
#print(feature_map)
for digit in range(10):
    #print(digit, len(feature_map[digit]))
    for f in feature_map[digit]:
        train.append(f)
        train_labels.append(digit)
print(int(time.time() - start), "seconds.")


# Train the SVM
print("Beginning training.", end=" ")
start = time.time()
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
#svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
#svm.train(np.array(train).astype(np.float32), cv2.ml.ROW_SAMPLE, np.array(train_labels))
svm.trainAuto(np.array(train).astype(np.float32), cv2.ml.ROW_SAMPLE, np.array(train_labels))
print(int(time.time() - start), "seconds.")

# Test the SVM
print("Beginning testing.", end=" ")
start = time.time()
data = read_images("data/mnist_medium_test.csv")
digit_map = make_digit_map(data)
feature_map = build_feature_map(digit_map, features)
test = []
test_labels = []
test_images = []
for digit in range(10):
    img_idx = 0
    for f in feature_map[digit]:
        test.append(f)
        test_labels.append(digit)
        test_images.append(digit_map[digit][img_idx]) # For viewing
        img_idx += 1
predictions = svm.predict(np.array(test).astype(np.float32))[1]

print(int(time.time() - start), "seconds.")
print( "R Test\n", np.array(testR(test_labels, predictions)))
print("AMD Test", testAMD(test_labels, predictions))

cv2.destroyAllWindows()
