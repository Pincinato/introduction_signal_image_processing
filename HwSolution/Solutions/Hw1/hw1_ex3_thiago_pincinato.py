import sys
import glob
import os
import numpy as np
import pdb
import matplotlib.pyplot as plt
# from scipy import misc
import random


def L1norm(a,b):

    x = 0
    y = 0
    if a[0] > b[0]:
        x= a[0] - b[0]
    else:
        x= b[0] - a[0]
    if a[1] > b[1]:
        y = a[1] - b[1]
    else:
        y = b[1] - a[1]
    return x+y


def getDistanceQAL(imageDistance):
    results = np.zeros(5)
    for i in range(1, imageDistance.shape[0]-1):
        for j in range(1, imageDistance.shape[1]):
            q =([i-1, j-1], [i-1, j], [i, j-1], [i, j], [i+1, j])
            for index in range(0, 5):
                results[index] = L1norm([i, j],q[index]) + imageDistance[q[index][0], q[index][1]]
            imageDistance[i][j] = min(results)
    return imageDistance


def getDistanceQBR(imageDistance):
    results = np.zeros(5)
    for i in range(imageDistance.shape[0]-2, -1, -1):
        for j in range(imageDistance.shape[1]-2, -1, -1):
            q = ([i-1, j+1], [i, j+1], [i+1, j+1], [i, j], [i+1, j])
            for index in range(0, 5):
                results[index] = L1norm([i, j],q[index]) + imageDistance[q[index][0], q[index][1]]
            imageDistance[i][j] = min(results)
    return imageDistance


def correctBorders(imageDistance):
    last_col = imageDistance.shape[0]-1
    last_row = imageDistance.shape[1]-1
    for i in range(imageDistance.shape[0]-2, -1, -1):
        imageDistance[i][last_col] = min([L1norm([i, last_col],[i, last_col-1]) + imageDistance[i][last_col - 1], imageDistance[i][last_col]])
    for j in range(imageDistance.shape[1]-2, -1, -1):
        imageDistance[last_row][j] = min([L1norm([last_row, j], [last_row - 1, j]) + imageDistance[last_row - 1][j], imageDistance[last_col][j]])
    imageDistance[last_row][last_col] = L1norm([last_row, last_col], [last_row -1 , last_col -1 ]) + imageDistance[last_row  -1][last_col -1]
    return imageDistance
# load shapes

shapes = glob.glob(os.path.join('shapes', '*.png'))
for i, shape in enumerate(shapes):
    # load the edge map
    edge_map = plt.imread(shape)
    # caclulate distance map
    # distance_map: array_like, same size as edge_map
    distance_map = L1norm([0, 0], [edge_map.shape[0], edge_map.shape[1]])*np.ones(edge_map.shape)
    distance_map[edge_map == 1] = 0
    distance_map = getDistanceQAL(distance_map)
    distance_map = getDistanceQBR(distance_map)
    distance_map = correctBorders(distance_map)
    distance_map = 1 - distance_map / np.amax(distance_map)
    # the top row of the plots should be the edge maps, and on the bottom the corresponding distance maps
    k, l = i+1, i+1+len(shapes)
    plt.subplot(2, len(shapes), k)
    plt.imshow(edge_map, cmap='gray')
    plt.subplot(2, len(shapes), l)
    plt.imshow(distance_map, cmap='gray')

plt.show()
