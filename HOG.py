# Histogram of Oriented Gradients
import numpy as np
import cv2

class HOG:
    kernels = None
    
    def __init__(self, H, W, patch, block, angle):
        #img_orig = cv2.imread("images/0_28x28.png", cv2.IMREAD_GRAYSCALE)
        #img_orig = cv2.resize(cv2.imread("images/sunriseCross.jpg", cv2.IMREAD_GRAYSCALE), (96, 64))
        #cv2.imshow("Original", img_orig)
        #cv2.waitKey()
        print("Init")
        self.H, self.W = H, W
        #print("The shape is ", img_orig.shape)
        self.patch_size = patch
        self.block_size = block
        self.angle_step = angle
        self.bins = [[[0]*(180//self.angle_step) for j in range(self.W//self.patch_size)]\
                for i in range(self.H//self.patch_size)]
        #print("bin shape:", len(self.bins), len(self.bins[0]), len(self.bins[0][0]))


    # Build HOG kernels
    # Builds the convolution kernels for the various gradient directions
    def build_kernels(self):
        N = 3;
        self.kernels = []
        for a in range(0, 180, self.angle_step):
            kernel = np.zeros((N, N))
            angle = a * np.pi/180.0
            for i in range(N):
                for j in range(N):
                    entry = (j-(N-1.0)/2)*np.cos(angle) + (i - (N-1.0)/2)*np.sin(angle)
                    kernel[i][j] = entry
                    
            self.kernels.append(kernel)
        """
        for k in self.kernels:
            print(k)
            print("")
        """

    """
    Compute the histogram of gradients for an image, using kernels

    img The image on which to compute the HOG
    block_size The size of a block for which to equalize the histogram
    patch_size The size of a patch in the image to convolve with
    kernels. This should be strictly smaller than block_size.
    """
    def computeHOG(self, img):
        self.img = img # Remember the image

        # clear out the bins
        self.bins = [[[0]*(180//self.angle_step) for j in range(self.W//self.patch_size)]\
                for i in range(self.H//self.patch_size)]
        #print("len", len(img))
        #print("len0", len(img[0]))
        #print(img.shape)
        if self.kernels == None:
            self.build_kernels()
        feature_vector = [] # Return value
        # Compute the HOG for each block
        for j in range(0, self.W - self.block_size + 1, self.patch_size): 
            for i in range(0, self.H - self.block_size + 1, self.patch_size):
                # Equalize histogram, copying to a new image
                new_img = cv2.equalizeHist(img[i:i + self.block_size, j:j + self.block_size])
                self.computeHOG_block(new_img, feature_vector, i, j)
        return feature_vector

    """
    Compute the histogram of gradients for an image, using kernels
    Adds the numbers to the feature_vector.
    img The image on which to compute the HOG
    patch_size The size of a patch in the image to convolve with kernels
    """
    def computeHOG_block(self, img, feature_vector, i, j):
        h, w = img.shape
        for k in range(len(self.kernels)):
            kernel = self.kernels[k]
            img_f = np.array(img, dtype='float32')
            im = cv2.filter2D(img_f, -1, kernel)
            #cv2.imshow("block", im)
            #cv2.waitKey()
            # Get feature vector for this kernel
            for jj in range(0, w - self.patch_size + 1, self.patch_size):
                for ii in range(0, h - self.patch_size + 1, self.patch_size):
                    summ = np.sum(im[ii:ii + self.patch_size, jj:jj + self.patch_size])
                    self.bins[(i+ii)//self.patch_size][(j+jj)//self.patch_size][k] += np.abs(summ)
                    feature_vector.append(np.abs(summ))
            

    # If the image has been binned, then this method will display the image, together
    # with some gradient lines
    def display_HOG(self):
        # We scale it so that the image's max dimension is 800
        h, w = self.img.shape
        #print(img.shape)
        scale = 800/h if h > w else 800/w
        img_hog = cv2.cvtColor(cv2.resize(self.img, (int(w*scale), int(h*scale))), cv2.COLOR_GRAY2BGR)
        cv2.imshow("scale", img_hog)
        
        # Put lines on the image
        line_length = 30
        m = np.max(self.bins)
        for i in range(h//self.patch_size):
            for j in range(w//self.patch_size):
                cx = scale*self.patch_size*(0.5 + j)
                cy = scale*self.patch_size*(0.5 + i)
                b = self.bins[i][j]
                for a in range(len(b)): 
                    length = line_length * b[a] / m if m != 0 else 0
                    angle = a * self.angle_step * np.pi / 180
                    x0 = int(cx - length * np.cos(angle))
                    x1 = int(cx + length * np.cos(angle))
                    y0 = int(cy - length * np.sin(angle))
                    y1 = int(cy + length * np.sin(angle))
                    cv2.line(img_hog, (x0, y0), (x1, y1), (255, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("lines", img_hog)
        cv2.waitKey()
        cv2.destroyAllWindows()


    def enhance_writing():
        img_orig = cv2.imread("images/Sawyer.JPG", cv2.IMREAD_GRAYSCALE)
        img_orig = cv2.imread("sudoku_square/sudoku72.png", cv2.IMREAD_GRAYSCALE)
        img = cv2.adaptiveThreshold(img_orig, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 3)
        img = img & 255 - (255 - img_orig)//3 # all 0 pixels in img force 0s in the and-ed image
        #img = img & img_orig
        #img = cv2.pyrDown(img)
        #img = cv2.pyrDown(img)
        #img_orig = cv2.pyrDown(img_orig)
        #img_orig = cv2.pyrDown(img_orig)
        cv2.imshow("Sawyer", img)
        cv2.imshow("Orig", img_orig) 
        cv2.waitKey()
        cv2.destroyAllWindows()


