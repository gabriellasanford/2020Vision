import numpy as np
import cv2
from matplotlib import pyplot as plt

# Feature set containing (x,y) values of 25 known/training data
#trainData = np.random.randint(0,100,(25,2)).astype(np.float32)

# Labels each one either Red or Blue with numbers 0 and 1
#responses = np.random.randint(0,2,(25,1)).astype(np.float32)
train = np.array([[1, 1], [1, 3], [1, 5], [3, 1], [3, 3], [3, 5], [5, 1], [5, 3], [5, 5],\
                  [12, 12], [12, 14], [14, 12], [14, 14]]).astype(np.float32)
labels = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1],\
                   [0], [0], [0], [0]]).astype(np.float32)

# Take Red families and plot them
red = train[labels.ravel()==0]
plt.scatter(red[:,0],red[:,1],80,'r','^')

# Take Blue families and plot them
blue = train[labels.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],80,'b','s')

# Green is the unknown data
green = np.array([[2, 2], [8, 9], [13, 13]]).astype(np.float32)
plt.scatter(green[:,0],green[:,1],80,'g','o')

print(train)
print(labels)

knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, labels)

unkn = np.array([[2, 2], [8, 9], [13, 13]]).astype(np.float32)
unkn = np.array([[2, 2]]).astype(np.float32)

ret, results, neighbors ,dist = knn.findNearest(unkn, 3)
print(ret)
print(results)
print(neighbors)
print(dist)


plt.show()
