import cv2
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


# im is the target image
# k is the kernel
# returns the convolution image, without reversing k
def convolve(im, k):
    kh, kw = k.shape
    imh, imw = im.shape
    im_w_border = np.zeros((kh + imh - 1, kw + imw -1))
    im_w_border[(kh-1)//2:(kh-1)//2+imh, (kw-1)//2:(kw-1)//2+imw] += im
    new_img = np.array([[np.sum(k*im_w_border[i:i+kh, j:j+kw]) \
                for j in range(imw)] for i in range(imh)], dtype='float')
    new_img[new_img>255] = 255
    new_img[new_img<0] = 0
    
    return new_img



"""
    TESTS
"""
# feature_map maps digits to list of list of features
# com_map maps digits to a com of that digit's features
def testR(labels, pred):
    predictions = [[0 for i in range(10)] for j in range(10)]
    for i in range(len(labels)):
        predictions[labels[i]][int(pred[i])] += 1
    return predictions


# feature_map maps digits to list of list of features
# com_map maps digits to a com of that digit's features
def testAMD(labels, pred):
    countr = 0.0
    countw = 0.0
    for i in range(len(labels)):
        if labels[i] == int(pred[i]):
            countr += 1
        else:
            countw += 1
    return countr/(countr +  countw)


"""
    TRAINING AND TESTING
"""
# Select the set of features
features = [meth.waviness, meth.slantiness]

# # Set up training data
# print("Extracting Features.", end=" ")
# start = time.time()
# data = read_images("data/mnist_train.csv")
# digit_map = make_digit_map(data)
# feature_map = build_feature_map(digit_map, features)
# train = []
# train_labels = []
# #print(feature_map)
# for digit in range(10):
#     #print(digit, len(feature_map[digit]))
#     for f in feature_map[digit]:
#         train.append(f)
#         train_labels.append(digit)
# print(int(time.time() - start), "seconds.")

# # Train the SVM
# print("Beginning training.", end=" ")
# start = time.time()
# svm = cv2.ml.SVM_create()
# #svm.setType(cv2.ml.SVM_C_SVC)
# #svm.setKernel(cv2.ml.SVM_LINEAR)
# svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
# #svm.train(np.array(train).astype(np.float32), cv2.ml.ROW_SAMPLE, np.array(train_labels))
# svm.trainAuto(np.array(train).astype(np.float32), cv2.ml.ROW_SAMPLE, np.array(train_labels))
# print(int(time.time() - start), "seconds.")

# Read SVM from file (comment out if training)
svm = cv2.ml_SVM.load("data/wav-slant-lrg-lrg.svm")

# Test the SVM
print("Beginning testing.", end=" ")
start = time.time()
data = read_images("data/mnist_test.csv")
digit_map = make_digit_map(data)
feature_map = build_feature_map(digit_map, features)
test = []
test_labels = []
for digit in range(10):
    for f in feature_map[digit]:
        test.append(f)
        test_labels.append(digit)
predictions = svm.predict(np.array(test).astype(np.float32))[1]

print(int(time.time() - start), "seconds.")
print( "R Test\n", np.array(testR(test_labels, predictions)))
print("AMD Test", testAMD(test_labels, predictions))
#for i in range(len(test)):
#    print(i, test_labels[i], predictions[i])


