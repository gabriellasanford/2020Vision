"""
* Read images
* Extract features from each image
* Find center of mass of features by label
* Classify images by nearest center of mass
* Test accuracy of classification on training data
* Test accuracy on test data
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import methods as meth

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


# Find center of mass of each digit's feature vectors
# feature_map is a map from each digit to a list of feature vectors
# Returns a map: digit -> Center of mass
def find_coms(feature_map):
    num_centers = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    com_map = {}
    for digit in feature_map:
        feature_matrix = np.asfarray(feature_map[digit])
        #com_map[digit] = np.mean(feature_matrix, axis=0)
        ret,label,centers = cv2.kmeans(np.asarray(feature_map[digit], dtype='float32'), num_centers, \
                    None, criteria, 10,cv2.KMEANS_RANDOM_CENTERS)
        com_map[digit] = centers
    return com_map
    

"""
    TESTS
"""
# Anthony/Michael A./David
# feature_map maps digits to list of list of features
# com_map maps digits to a com of that digit's features
# Returns the fraction of correctly-classified images
def testAMD(feature_map, com_map):
    ccom = None
    countr = 0.0
    countw = 0.0
    for key in feature_map:
        for ele in feature_map[key]:
            mdist = float("inf")
            for ckey in com_map:
                for com in com_map[ckey]:
                    dist = abs(np.linalg.norm(com-ele))
                    if dist < mdist:
                        mdist = dist
                        ccom = ckey
            if ccom == key:
                countr +=1  
            else:
                countw +=1
    per = countr/(countr +  countw)
    return per 


# Sri / Paul / Michael Bujard
# Test
# feature_map maps digits to list of list of features
# com_map maps digits to a com of that digit's features
def testSPM(feature_map, com_map):
    num_correct = 0.0
    num_total = 0.0
    for correct_digit, list_of_features in feature_map.items():
        for some_features in list_of_features:
            smallest_distance = float("inf")
            guess_digit = None
            for candidate_digit, com_list in com_map.items():
                for a_com in com_list:
                    distance = np.linalg.norm(a_com - some_features)
                    if distance < smallest_distance:
                        smallest_distance = distance
                        guess_digit = candidate_digit
            if guess_digit == correct_digit:
                num_correct += 1
            num_total += 1
    return num_correct / num_total


# Rob Hochberg
# feature_map maps digits to list of list of features
# com_map maps digits to a com of that digit's features
def testR(feature_map, com_map):
    predictions = [[0 for i in range(10)] for j in range(10)]
    for correct_digit, list_of_features in feature_map.items():
        for some_features in list_of_features:
            smallest_distance = float("inf")
            guess_digit = None
            for candidate_digit, com_list in com_map.items():
                for a_com in com_list:
                    distance = np.linalg.norm(a_com - some_features)
                    if distance < smallest_distance:
                        smallest_distance = distance
                        guess_digit = candidate_digit
            predictions[correct_digit][guess_digit] += 1
    return predictions

# Tests a feature on the traninig data.
def testKnn(feature):
    feature_map = build_feature_map(test_digit_map, [feature])
    knn = getTrainedKnn(feature)

    success = 0.0
    failure = 0.0
    predictions = [[0 for i in range(10)] for j in range(10)]
    for digit in range(10):
        for f in feature_map[digit]:
            unkn = np.array([f]).astype(np.float32)
            ret, results, neighbors ,dist = knn.findNearest(unkn, 3)
            prediction = results[0][0]
            predictions[digit][int(prediction)] += 1
            if prediction == digit:
                success += 1
            else:
                failure += 1
    print(success/(success + failure))
    print(np.array(predictions))

# List of implemented feature functions
all_features = [meth.draw_and_quarter, meth.waviness, meth.hv_weights,\
                meth.top_bottom_balance, meth.combineWavy,\
                meth.vertical_lines, meth.sectional_density, meth.slantiness,\
                meth.edginess, meth.Sobelness]

# Returns a trained KNN object for a feature.
def getTrainedKnn(feature):
    print ("\n\n", feature.__name__)

    # train
    feature_map = build_feature_map(training_digit_map, [feature])

    train = []
    labels = []
    for digit in range(10):
        for f in feature_map[digit]:
            train.append(f)
            labels.append(digit)
    #print(train)
    #print(labels)
    
    knn = cv2.ml.KNearest_create()
    knn.train(np.array(train).astype(np.float32), cv2.ml.ROW_SAMPLE, np.array(labels).astype(np.float32))

    return knn

# Sri, Anthony, Eniola
# classifies a single image given a knn and the feature map of the image
def classify_digit(knn, feature_map):
    _, results, _, _ = knn.findNearest(np.array([feature_map]).astype(np.float32), 3)
    prediction = results[0][0]
    return prediction

def test_existing_knn(feature, knn, test_map):
    feature_map = build_feature_map(test_map, [feature])

    success = 0.0
    failure = 0.0
    predictions = [[0 for i in range(10)] for j in range(10)]
    for digit in range(10):
        for f in feature_map[digit]:
            unkn = np.array([f]).astype(np.float32)
            ret, results, neighbors ,dist = knn.findNearest(unkn, 3)
            prediction = results[0][0]
            predictions[digit][int(prediction)] += 1
            if prediction == digit:
                success += 1
            else:
                failure += 1
    print(success/(success + failure))
    print(np.array(predictions))

# Returns a trained KNN object for a feature.
def make_trained_knn(feature, training_map):
    print ("\n\n", feature.__name__)
    
    # train
    feature_map = build_feature_map(training_map, [feature])

    train = []
    labels = []
    for digit in range(10):
        for f in feature_map[digit]:
            train.append(f)
            labels.append(digit)
    #print(train)
    #print(labels)
    
    knn = cv2.ml.KNearest_create()
    knn.train(np.array(train).astype(np.float32), cv2.ml.ROW_SAMPLE, np.array(labels).astype(np.float32))

    return knn
    
training_data = read_images("data/mnist_medium_train.csv")
training_digit_map = make_digit_map(training_data)

test_data = read_images("data/mnist_medium_test.csv")
test_digit_map = make_digit_map(test_data)

# List of implemented feature functions
all_features = [meth.draw_and_quarter, meth.waviness, meth.hv_weights,\
                meth.top_bottom_balance, meth.combineWavy,\
                meth.vertical_lines, meth.sectional_density, meth.slantiness,\
                meth.edginess, meth.Sobelness]
all_features = []

for f in all_features:
    testKnn(f)

    
    
    


           
