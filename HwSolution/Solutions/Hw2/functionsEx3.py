import scipy
import numpy as np
import functionsEx1 as pinc

# 3.1
# Write a function myharris(image) which computes the harris corner for each pixel in the image. The function should return the R
# response at each location of the image.
# HINT: You may have to play with different parameters to have appropriate R maps.
# Try Gaussian smoothing with sigma=0.2, Gradient summing over a 5x5 region around each pixel and k = 0.1.)
def myharris(image, w_size, sigma, k):
    # This function computes the harris corner for each pixel in the image
    # INPUTS
    # @image    : a 2-D image as a numpy array
    # @w_size   : an integer denoting the size of the window over which the gradients will be summed
    # sigma     : gaussian smoothing sigma parameter
    # k         : harris corner constant
    # OUTPUTS
    # @R        : 2-D numpy array of same size as image, containing the R response for each image location

    dx = np.array([[-1, 0, 1]])
    dy = dx.T
    # getting gaussian 2d filter
    g = pinc.gauss2d(sigma, w_size)
    # acquiring gdx and gdy
    gdx = scipy.signal.convolve(g, dx, mode='same', method='auto')
    gdy = scipy.signal.convolve(g, dy, mode='same', method='auto')
    # obtaining Ix, Iy, Iyy, Ixy and Ixx
    Ix = scipy.signal.convolve(image, gdx, mode='same', method='auto')
    Iy = scipy.signal.convolve(image, gdy, mode='same', method='auto')
    Ixx = np.multiply(Ix, Ix)
    Iyy = np.multiply(Iy, Iy)
    Ixy = np.multiply(Ix, Iy)
    # creating matrix \1\
    I1 = np.ones([w_size,w_size])
    # applying \1\*Iab to obtain Sab
    Sxx = scipy.signal.convolve(Ixx, I1, mode='same', method='auto')
    Sxy = scipy.signal.convolve(Ixy, I1, mode='same', method='auto')
    Syy = scipy.signal.convolve(Iyy, I1, mode='same', method='auto')
    H = np.zeros([2, 2])
    # creation of R matrix
    R = np.zeros([image.shape[0],image.shape[1]])
    # computing R value for each pixel of the original image
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            H = np.reshape([Sxx[y, x], Sxy[y, x], Sxy[y, x], Syy[y, x]], (2, 2))
            R[y, x] = np.linalg.det(H) - k*np.trace(H)*np.trace(H)
    return R

