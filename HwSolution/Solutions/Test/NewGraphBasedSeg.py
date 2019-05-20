import argparse
import logging
import time
from graph import build_graph, segment_graph
from random import random
from PIL import Image, ImageFilter
from skimage import io
import numpy as np
import scipy
import matplotlib.pyplot as plt


def diff(img, x1, y1, x2, y2):
    _out = np.sum((img[x1, y1] - img[x2, y2]) ** 2)
    return np.sqrt(_out)


def threshold(size, const):
    return (const * 1.0 / size)


def generate_image(forest, width, height):
    random_color = lambda: (int(random() * 255), int(random() * 255), int(random() * 255))
    colors = [random_color() for i in range(width * height)]
    img = Image.new('RGB', (width, height))
    im = img.load()
    for y in range(height):
        for x in range(width):
            comp = forest.find(y * width + x)
            im[x, y] = colors[comp]

    return img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)


def get_segmented_image(input_img, sigma=1.0, neighbor=8, K=10.0, min_comp_size=200):

    size = input_img.shape  # (width, height) in Pillow/PIL
    # Gaussian Filter
    smooth = scipy.ndimage.gaussian_filter(input_img, sigma)
    #smooth = input_img.filter(ImageFilter.GaussianBlur(sigma))
    smooth = np.array(smooth)
    graph_edges = build_graph(smooth, size[0], size[1], diff, neighbor == 8)
    forest = segment_graph(graph_edges, size[0] * size[1], K, min_comp_size, threshold)
    out_image = generate_image(forest, size[0], size[1])
    return  out_image
