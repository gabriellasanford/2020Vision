# Histogram of Oriented Gradients
import numpy as np
import cv2

# Build HOG kernels
N = 3;
kernels = []
angleStep = 30;
for a in range(0, 180, angleStep):
    kernel = np.zeros((N, N))
    angle = a * np.pi/180.0;
    for i in range(N):
        for j in range(N):
            entry = np.sin(np.pi/2.0*((j-(N-1.0)/2)*np.cos(angle)
                            + (i - (N-1.0)/2)*np.sin(angle)));
            kernel[i][j] = entry
            
    kernels.append(kernel);

for k in kernels:
    print(k)

