""" 2 Finding edges """

import numpy as np
from skimage import color, io
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

import functionsEx2 as ex2
import functionsEx1 as pinc

# load image
img = io.imread('bird.jpg')
img = color.rgb2gray(img)

# 2.1
# Gradients
# define a derivative operator
dx = np.array([[-1, 0, 1]])
dy = np.array([[-1, 0, 1]])
# convolve derivative operator with a 1d gaussian filter with sigma = 1
# You should end up with 2 1d edge filters,  one identifying edges in the x direction, and
# the other in the y direction
sigma = 1
gaussianFilter = pinc.gauss1d(sigma, 3)
gdx = np.array(pinc.myconv2(gaussianFilter, dx))
gdy = np.transpose(pinc.myconv2(gaussianFilter, dy))
# 2.2
# Gradient Edge Magnitude Map


# create an edge magnitude image using the derivative operator
circle = plt.imread('circle.jpg')
img_edge_mag, img_edge_dir = ex2.create_edge_magn_image(img, gdx, gdy)
img_edge_mag_circle, img_edge_dir_circle = ex2.create_edge_magn_image(circle, gdx, gdy)
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

# verify with circle image
circle = plt.imread('circle.jpg')
edge_maps_circles = ex2.make_edge_map(circle, gdx, gdy)
edge_maps_in_row = [edge_maps_circles[:, :, i] for i in range(edge_maps_circles.shape[2])]
all_in_row = np.concatenate((edge_maps_in_row), axis=1)
plt.imshow(np.concatenate((circle, all_in_row), axis=1))
plt.title('Circle and edge orientations')
plt.show()

# now try with original image
edge_maps = ex2.make_edge_map(img, gdx, gdy)
edge_maps_in_row = [edge_maps[:, :, i] for i in range(edge_maps.shape[2])]
all_in_row = np.concatenate((edge_maps_in_row), axis=1)
plt.imshow(np.concatenate((img, all_in_row), axis=1))
plt.title('Original image and edge orientations')
plt.show()


# 2.4

# show the result
img_non_max_sup = ex2.edge_non_max_suppression(img_edge_mag, edge_maps)
img_non_max_sup_circle = ex2.edge_non_max_suppression(img_edge_mag_circle, edge_maps_circles)
plt.figure(1)
plt.subplot(1, 3, 1)
plt.imshow(circle)
plt.title('Original image')
plt.subplot(1, 3, 2)
plt.imshow(img_edge_mag_circle)
plt.title('magnitude edge')
plt.subplot(1, 3, 3)
plt.imshow(img_non_max_sup_circle)
plt.title('max suppresion')

plt.figure(2)
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title('Original image')
plt.subplot(1, 3, 2)
plt.imshow(img_edge_mag)
plt.title('magnitude edge')
plt.subplot(1, 3, 3)
plt.imshow(img_non_max_sup)
plt.title('max suppresion')

plt.show()

