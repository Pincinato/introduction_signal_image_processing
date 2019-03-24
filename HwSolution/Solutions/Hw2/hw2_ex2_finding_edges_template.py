""" 2 Finding edges """

import numpy as np
from skimage import color, io
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import pdb

# load image
img = io.imread('bird.jpg')
img = color.rgb2gray(img)


### copy functions myconv2, gauss1d, gauss2d and gconv from exercise 1 ###


# 2.1
# Gradients
# define a derivative operator
dx = ### your code should go here ###
dy = ### your code should go here ###

# convolve derivative operator with a 1d gaussian filter with sigma = 1
# You should end up with 2 1d edge filters,  one identifying edges in the x direction, and
# the other in the y direction
sigma = 1
### your code should go here ###
gdx = ### your code should go here ###
gdy = ### your code should go here ###


# 2.2
# Gradient Edge Magnitude Map
def create_edge_magn_image(image, dx, dy):
    # this function created an eddge magnitude map of an image
    # for every pixel in the image, it assigns the magnitude of gradients
    # INPUTS
    # @image  : a 2D image
    # @gdx     : gradient along x axis
    # @gdy     : geadient along y axis
    # OUTPUTS
    # @ grad_mag_image  : 2d image same size as image, with the magnitude of gradients in every pixel
    # @grad_dir_image   : 2d image same size as image, with the direcrion of gradients in every pixel

    ### your code should go here ###

    return grad_mag_image, grad_dir_image


# create an edge magnitude image using the derivative operator
img_edge_mag, img_edge_dir = create_edge_magn_image(img, dx, dy)

# show all together
plt.subplot(121)
plt.imshow(img)
plt.axis('off')
plt.title('Original image')
plt.subplot(122)
plt.imshow(img_edge_mag)
plt.axis('off')
plt.title('Edge magnitude map')
plt.show()

# 2.3
# Edge images of particular directions
def make_edge_map(image, dx, dy):
    # INPUTS
    # @image        : a 2D image
    # @gdx          : gradient along x axis
    # @gdy          : geadient along y axis
    # OUTPUTS:
    # @ edge maps   : a 3D array of shape (image.shape, 8) containing the edge maps on 8 orientations

    ### your code should go here ###

    return edge_maps


# verify with circle image
circle = plt.imread('circle.jpg')
edge_maps = make_edge_map(circle, dx, dy)
edge_maps_in_row = [edge_maps[:, :, i] for i in range(edge_maps.shape[2])]
all_in_row = np.concatenate((edge_maps_in_row), axis=1)
plt.imshow(np.concatenate((circle, all_in_row), axis=1))
plt.title('Circle and edge orientations')
# plt.imshow(np.concatenate(np.dsplit(edge_maps, edge_maps.shape[2]), axis=0))
plt.show()

# now try with original image
edge_maps = make_edge_map(img, dx, dy)
edge_maps_in_row = [edge_maps[:, :, i] for i in range(edge_maps.shape[2])]
all_in_row = np.concatenate((edge_maps_in_row), axis=1)
plt.imshow(np.concatenate((img, all_in_row), axis=1))
plt.title('Original image and edge orientations')
plt.show()


# 2.4
# Edge non max suppresion
def edge_non_max_suppression(img_edge_mag, edge_maps):
    # This function performs non maximum suppresion, in order to reduce the width of the edge response
    # INPUTS
    # @img_edge_mag   : 2d image, with the magnitude of gradients in every pixel
    # @edge_maps      : 3d image, with the edge maps
    # OUTPUTS
    # @non_max_sup    : 2d image with the non max suppresed pixels

    ### your code should go here ###

    return non_max_sup


# show the result
img_non_max_sup = edge_non_max_suppression(img_edge_mag, edge_maps)
plt.imshow(np.concatenate((img, img_edge_mag, img_non_max_sup), axis=1))
plt.title('Original image, magnitude edge, and max suppresion')
plt.show()


# 2.5
# Canny edge detection (BONUS)
def canny_edge(image, sigma=2):
    # implementation of canny edge detector
    # INPUTS
    # @image      : 2d image
    # @sigma      : sigma parameter of gaussian
    # OUTPUTS
    # @canny_img  : 2d image of size same as image, with the result of the canny edge detection

    ### your code should go here ###

    return canny_img

canny_img = canny_edge(img)
plt.imshow(np.concatenate((img, canny_img), axis=1))
plt.title('Original image and canny edge detector')
plt.show()
