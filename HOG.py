# Histogram of Oriented Gradients
import numpy as np
import cv2

# Build HOG kernels
def build_kernels():
    N = 3;
    kernels = []
    angleStep = 20;
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

    return kernels


"""
Compute the histogram of gradients for an image, using kernels

img The image on which to compute the HOG
block_size The size of a block for which to equalize the histogram
patch_size The size of a patch in the image to convolve with
kernels. This should be strictly smaller than block_size.
"""
def computeHOG(img, block_size, patch_size):
    print("len", len(img))
    print("len0", len(img[0]))
    print(img)
    print(img.shape)
    kernels = build_kernels()
    feature_vector = [] # Return value
    # Compute the HOG for each block
    W, H = img.shape
    for j in range(0, W - block_size + 1, patch_size): 
        for i in range(0, H - block_size + 1, patch_size):
            # Equalize histogram, copying to a new image
            new_img = cv2.equalizeHist(img[j:j + block_size, i:i + block_size])
            computeHOG_block(new_img, patch_size, feature_vector, kernels)
    return feature_vector

"""
Compute the histogram of gradients for an image, using kernels
Adds the numbers to the feature_vector.
img The image on which to compute the HOG
patch_size The size of a patch in the image to convolve with kernels
"""
def computeHOG_block(img, patch_size, feature_vector, kernels):
    W, H = img.shape
    for kernel in kernels:
        im = cv2.filter2D(img, -1, kernel)
        #print("New Im")
        #print(kernel)
        #print(img)
        #print(im)
        # Get feature vector for this kernel
        for j in range(0, W - patch_size + 1, patch_size):
            for i in range(0, H - patch_size + 1, patch_size):
                
                #print(im[i:i + patch_size, j:j + patch_size])
                sum = np.sum(im[j:j + patch_size, i:i + patch_size])
                feature_vector.append(sum)


def display_HOG(img, hog, block_size, patch_size):
    # We scale it so that the image's max dimension is 800
    W, H = img.shape
    print(img.shape)
    scale = 800/H if H > W else 800/W
    img_hog = cv2.cvtColor(cv2.resize(img, (int(W*scale), int(H*scale))), cv2.COLOR_GRAY2BGR)
    cv2.imshow("scale", img_hog)
    bins = [[[0]*9 for i in range(H//patch_size)] for j in range(W//patch_size)]
    idx = 0
    for j in range(0, W - block_size + 1, patch_size): 
        for i in range(0, H - block_size + 1, patch_size):
            for k in range(9):
                for jj in range(0, block_size - patch_size + 1, patch_size):
                    for ii in range(0, block_size - patch_size + 1, patch_size):
                        I = (i + ii)//patch_size
                        J = (j + jj)//patch_size
                        bins[J][I][k] += hog[idx]
                        idx += 1
    print(bins)                  

    # Put lines on the image
    line_length = 20
    print(H,W)
    for i in range(H//patch_size):
        for j in range(W//patch_size):
            cx = scale*patch_size*(0.5 + j)
            cy = scale*patch_size*(0.5 + i)
            b = bins[j][i]
            m = max(b)
            for a in range(9):
                length = line_length * b[a] / m if m != 0 else 0
                angle = a * np.pi / 9
                x0 = int(cx - length * np.cos(angle))
                x1 = int(cx + length * np.cos(angle))
                y0 = int(cy - length * np.sin(angle))
                y1 = int(cy + length * np.sin(angle))
                cv2.line(img_hog, (x0, y0), (x1, y1), (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow("lines", img_hog)
    cv2.waitKey()
    cv2.destroyAllWindows()

def enhance_writing():
    img_orig = cv2.imread("images/Sawyer.JPG", cv2.IMREAD_GRAYSCALE)
    #img_orig = cv2.imread("sudoku_square/sudoku72.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.adaptiveThreshold(img_orig, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 3)
    img = img & 255 - (255 - img_orig)//3
    #img = img & img_orig
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    img_orig = cv2.pyrDown(img_orig)
    img_orig = cv2.pyrDown(img_orig)
    cv2.imshow("Sawyer", img)
    cv2.imshow("Orig", img_orig)
    cv2.waitKey()
    cv2.destroyAllWindows()

#img_orig = cv2.imread("images/0_28x28.png", cv2.IMREAD_GRAYSCALE)
img_orig = cv2.resize(cv2.imread("images/sunrise.jpg", cv2.IMREAD_GRAYSCALE), (60,80))
#img_orig = cv2.resize(cv2.imread("images/sudoku25.png", cv2.IMREAD_GRAYSCALE), (60,80))
vec = computeHOG(img_orig, 20, 10)
print(vec)
display_HOG(img_orig, vec, 20, 10)
