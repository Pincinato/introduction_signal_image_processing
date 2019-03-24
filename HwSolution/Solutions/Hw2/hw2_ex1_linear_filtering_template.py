""" 1 Linear filtering """

# Imports
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import time

import functionsEx1 as ex1


figure_counter = 0
plt.figure(figure_counter)
img = plt.imread('cat.jpg').astype(np.float32)
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.axis('off')
plt.title('original image')


# 1.3
# create a boxfilter of size 10 and convolve this filter with your image - show the result
bsize = 11
img2 = ex1.myconv2(img, ex1.boxfilter(bsize))
plt.subplot(1, 2, 2)
plt.imshow(img2)
plt.axis('off')
plt.title('convoluted image (box filter)')



# Display a plot using sigma = 3
# run your gconv on the image for sigma=3 and display the result
sigma = 3

figure_counter = figure_counter+1
plt.figure(figure_counter)
plt.imshow(ex1.gauss2d(sigma, 11))
plt.title('2D Gaussian filter ')
figure_counter = figure_counter+1
plt.figure(figure_counter)
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.axis('off')
plt.title('original image')
plt.subplot(1, 2, 2)
plt.imshow(ex1.gconv(img, sigma))
plt.title('convoluted image (Gaussian filter)')
plt.show()
# 1.7
# Convolution with a 2D Gaussian filter is not the most efficient way
# to perform Gaussian convolution with an image. In a few sentences, explain how
# this could be implemented more efficiently and why this would be faster.
#
# HINT: How can we use 1D Gaussians?

### your explanation should go here ###

# One could apply the convolution of the image with teh 1D gauss filter and after a another convolution
# with the result of the first convolution and the transpose of the 1D gauss filter. The result would
# be exactly the same as applying the convolution with 2D gauss filter.
# The mathematical demonstration that the result would be the same can be seen in below:

# Note that this operation is possible due to the fact that 2D gauss filter are symmetric
# and can be separable.

# When doing the filtering process with 1D gauss filter and its transpose, we are realizing
# (m + m) multiplication by pixel instead of m*m (2D gauss filter). It means that we salve ((m*m) -(m+m))*nxn operations
# , and thus, we can enhance our efficiency and velocity.


# 1.8
# Computation time vs filter size experiment
size_range = np.arange(3, 100, 5)
t1d = np.zeros(size_range.shape)
t2d = np.zeros(size_range.shape)
index = 0
for size in size_range:
    # Building filters
    filter_box = ex1.boxfilter(size)
    flt1d = np.array([filter_box[0, :]])
    flt1dT = flt1d.T
    # measuring time 2d filter
    start = time.time()
    ex1.myconv2(img, filter_box)
    t2d[index] = time.time() - start
    # measuring time 1d filter
    start = time.time()
    ex1.myconv2(ex1.myconv2(img, flt1d),flt1dT)
    t1d[index] = time.time()- start
    index = index +1

# plot the comparison of the time needed for each of the two convolution cases
plt.plot(size_range, t1d, label='1D filtering')
plt.plot(size_range, t2d, label='2D filtering')
plt.xlabel('Filter size')
plt.ylabel('Computation time')
plt.legend(loc=0)
plt.show()
