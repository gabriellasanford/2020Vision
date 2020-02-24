import numpy as np
import matplotlib.pyplot as plt


image_size = 28

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
def find_com(feature_map):
    com_map = {}
    for digit in feature_map:
        feature_matrix = np.asfarray(feature_map[digit])
        com_map[digit] = np.mean(feature_matrix, axis=0)
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
                dist = abs(np.linalg.norm(com_map[ckey]-ele))
                if dist <= mdist:
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
            for candidate_digit, a_com in com_map.items():
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
            for candidate_digit, a_com in com_map.items():
                distance = np.linalg.norm(a_com - some_features)
                if distance < smallest_distance:
                    smallest_distance = distance
                    guess_digit = candidate_digit
            predictions[correct_digit][guess_digit] += 1
    return predictions

def test_features(features_list):
    data = read_images("data/mnist_medium.csv")
    digit_map = make_digit_map(data)

    result_map = {}

    for f in features_list:
        print ("\n\nTesting: ", f.__name__)

        features = [f]
        
        # train
        feature_map = build_feature_map(digit_map, features)
        com = find_com(feature_map)

        # Test on training data
        amd_test_on_training_data = testAMD(feature_map, com)
        spm_test_on_training_data = testSPM(feature_map, com)
        prediction_matrix_on_training_data = testR(feature_map, com)

        # Test on test data
        data = read_images("data/mnist_medium_test.csv")
        digit_map = make_digit_map(data)
        feature_map = build_feature_map(digit_map, features)

        amd_test_on_test_data = testAMD(feature_map, com)
        spm_test_on_test_data = testSPM(feature_map, com)
        prediction_matrix_on_test_data = testR(feature_map, com)

        digit_prediction_success_rates = []
        for digit in range(10):
            prediction_for_digit = prediction_matrix_on_test_data[digit]
            correct_predictions = prediction_for_digit[digit]
            total_test_data_size = sum(prediction_for_digit)
            success_rate = correct_predictions / total_test_data_size
            digit_prediction_success_rates.append((digit, success_rate))
        
        digit_prediction_success_rates.sort(key=lambda x: x[1])
        result_map[f] = (amd_test_on_training_data, spm_test_on_training_data, prediction_matrix_on_training_data, \
            amd_test_on_test_data, spm_test_on_test_data, prediction_matrix_on_test_data, \
                digit_prediction_success_rates)
        
    return result_map

def get_most_successful_predictions(test_result, k):
    return test_result[6][-k:]

def get_most_unsuccessful_predictions(test_result, k):
    return test_result[6][0:k]

def get_prediction_matrix_on_training_data(test_result):
    return test_result[2]

def get_prediction_matrix_on_test_data(test_result):
    return test_result[5]

def get_training_data_set_success_rate(test_result):
    return test_result[0]

def get_test_data_set_success_rate(test_result):
    return test_result[3]

def sort_features_by_success_rate(features):
    test_result_map = test_features(features)
    return [(f.__name__, get_test_data_set_success_rate(test_result_map[f])) \
        for f in sorted(features, key=lambda x: get_test_data_set_success_rate(test_result_map[x]))]

def create_success_plots(features):
    test_result_map = test_features(features)
    feature_names = [f.__name__ for f in features]
    feature_successes = [get_test_data_set_success_rate(test_result_map[f]) \
        for f in sorted(features, key=lambda x: get_test_data_set_success_rate(test_result_map[x]))]

    y_pos = np.arange(len(feature_names))
    
    # Create bars
    plt.bar(y_pos, feature_successes)
    
    # Create names on the x-axis
    plt.xticks(y_pos, feature_names)
    
    # Show graphic
    plt.show()


