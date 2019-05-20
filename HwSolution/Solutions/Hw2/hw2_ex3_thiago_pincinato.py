""" 3 Corner detection """


# Imports
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
from scipy.signal import convolve2d, convolve
from skimage import color, io
import scipy

import functionsEx3 as ex3

# Load the image, convert to float and grayscale
img = io.imread('chessboard.jpeg')
img = color.rgb2gray(img)

# 3.1
# Write a function myharris(image) which computes the harris corner for each pixel in the image. The function should return the R
# response at each location of the image.

# In functionsEx3.py file

# 3.2
# Evaluate myharris on the image
R = ex3.myharris(img, 5, 0.2, 0.1)
plt.figure(1)
plt.imshow(img)
plt.title("Original image in gray scale")
plt.figure(2)
plt.imshow(R)
plt.colorbar()
plt.title("R values")
plt.figure(3)
thresholdInR = np.array(R)
thresholdInR[np.where(thresholdInR < 2)] = 0
thresholdInR[np.where(thresholdInR != 0)] = 255
plt.imshow(thresholdInR)
plt.title("R > 2")
plt.show()


# 3.3
# Repeat with rotated image by 45 degrees
# HINT: Use scipy.ndimage.rotate() function
R_rotated = ex3.myharris(scipy.ndimage.rotate(img, 45, reshape=False), 5, 0.2, 0.1)  ### your code should go here ###
plt.figure(1)
plt.imshow(scipy.ndimage.rotate(img, 45, reshape=False))
plt.title("Image in gray scale rotated")
plt.figure(2)
plt.imshow(R_rotated)
plt.colorbar()
plt.title("R values")
plt.figure(3)
thresholdInR = np.array(R_rotated)
thresholdInR[np.where(thresholdInR < 2)] = 0
thresholdInR[np.where(thresholdInR != 0)] = 255
plt.imshow(thresholdInR)
plt.title("R > 2")
plt.show()


# 3.4
# Repeat with downscaled image by a factor of half
# HINT: Use scipy.misc.imresize() function
R_scaled = ex3.myharris(scipy.misc.imresize(img, size=50), 5, 0.2, 0.1)
plt.figure(1)
plt.imshow(scipy.misc.imresize(img,  size=50))
plt.title("Downscaled image by half")
plt.figure(2)
plt.imshow(R_scaled)
plt.colorbar()
plt.title("R values")
plt.figure(3)
thresholdInR = np.array(R_scaled)
thresholdInR[np.where(thresholdInR < 2000000000)] = 0
thresholdInR[np.where(thresholdInR != 0)] = 255
plt.imshow(thresholdInR)
plt.title("R > 2000000000")
plt.show()

