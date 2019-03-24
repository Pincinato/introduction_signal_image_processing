

import numpy as np

# 1.1
def boxfilter(n):
    # this function returns a box filter of size nxn

    ### generating filter ###
    box_filter = np.ones((n, n)) / (n*n)
    return box_filter


# 1.2
# Implement full convolution
def myconv2(image, filt):
    # This function performs a 2D convolution between image and filt, image being a 2D image. This
    # function should return the result of a 2D convolution of these two images. DO
    # NOT USE THE BUILT IN SCIPY CONVOLVE within this function. You should code your own version of the
    # convolution, valid for both 2D and 1D filters.
    # INPUTS
    # @ image         : 2D image, as numpy array, size mxn
    # @ filt          : 1D or 2D filter of size kxl
    # OUTPUTS
    # img_filtered    : 2D filtered image, of size (m+k-1)x(n+l-1)

    (m, n) = np.shape(image)
    (k, l) = np.shape(filt)
    # padding image with 0
    img_padded = np.pad(image, [(k - 1, k - 1), (l - 1, l - 1)], 'constant', constant_values=(0, 0))
    # flipping filter
    flipped_filter = np.flip(filt)
    #appling convolution
    filtered_img = [[np.sum(np.multiply(img_padded[y:y + k, x:x + l], flipped_filter)) for x in range(n + l - 1)] for y in range(m + k - 1)]
    return filtered_img


# 1.4
# create a function returning a 1D gaussian kernel
def gauss1d(sigma, filter_length=10):
    # INPUTS
    # @ sigma         : sigma of gaussian distribution
    # @ filter_length : integer denoting the filter length, default is 10
    # OUTPUTS
    # @ gauss_filter  : 1D gaussian filter
    # correcting length if it is even
    if filter_length %2 == 0:
        filter_length= filter_length + 1
    # creating array x
    x = np.linspace(-(filter_length-1)/2,(filter_length-1)/2, filter_length);
    # generating gauss filter
    gauss_filter = np.exp(- np.power(x,2)/(2*sigma*sigma))
    # normalizing
    gauss_filter = np.array([gauss_filter/gauss_filter.sum()])
    return gauss_filter


# 1.5
# create a function returning a 2D gaussian kernel
def gauss2d(sigma, filter_size=10):
    # INPUTS
    # @ sigma         : sigma of gaussian distribution
    # @ filter_size   : integer denoting the filter size, default is 10
    # OUTPUTS
    # @ gauss2d_filter  : 2D gaussian filter

    # Generating 1D gauss
    gauss1d_filter=gauss1d(sigma,filter_size)
    # Convolution of 1D gauss and its transpose
    gauss2d_filter = myconv2(gauss1d_filter,gauss1d_filter.T)
    return gauss2d_filter


# 1.6
# Convoltion with gaussian filter
def gconv(image, sigma, filter_size=11):
    # INPUTS
    # image           : 2d image
    # @ sigma         : sigma of gaussian distribution
    # @ filter_size   : size of filter
    # OUTPUTS
    # @ img_filtered  : filtered image with gaussian filter
    img_filtered = myconv2(image, gauss2d(sigma, filter_size))
    return img_filtered
