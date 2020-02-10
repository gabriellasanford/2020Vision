import math
import numpy as np
shoe = [10, 10, 5, 5, 5]

data = [3, 4, 0, 0, 1, 10, 20, 5, 8, 12, 11, 6, 5, 6, 24, 9, 0, 0, \
        3, 7, 4, 8, 12, 19, 2, 11, 16, 4, 3, 12, 7, 0, 0, 0, 2, 2, 1, 1, 1]

def find_shoe(shoe, data):
    for i in range(len(data)-len(shoe) + 1):
        dp = 0
        leng = 0
        
        for j in range(5):
            dp += shoe[j] * data[i + j]
            leng += data[i + j] * data[i + j]
        dp /= math.sqrt(leng)
        
        region = data[i:i+5]
        dp = np.dot(shoe, region) / np.linalg.norm(region)
        print i, dp

            
    
find_shoe(shoe, data)
