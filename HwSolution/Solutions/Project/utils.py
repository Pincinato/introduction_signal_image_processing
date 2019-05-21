from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import random
import os.path
from scipy.misc import imread
import pdb


def extract_mask_middle_layers(img_gray_):
    img_to_work_ = np.multiply(img_gray_,255).astype('uint8')
    hist = np.histogram(img_to_work_, bins=np.arange(256))
    middle_layer_color = hist[1][np.argmax(hist[0])]
    found_mask = (img_to_work_ == middle_layer_color)
    return found_mask


def rgb_2_gray(img_):
    out_ = np.multiply(img_[:, :, 0], 0.3) + np.multiply(img_[:, :, 1], 0.59) + np.multiply(img_[:, :, 2], 0.11)
    out_ = np.multiply(out_, 25500).astype('uint16')
    return out_

def subplot(figure_number, suplot_number, plot_1, plot_2,title1, title2, op1, op2):
    plt.figure(figure_number)
    plt.subplot(2, 4, suplot_number)
    plt.title(title1)
    if op1 == 0:
        plt.imshow(plot_1)
    else:
        plt.imshow(plot_1,cmap='gray')
    plt.subplot(2, 4, suplot_number+1)
    plt.title(title2)
    if op2 == 0:
        plt.imshow(plot_2)
    else:
        plt.imshow(plot_2,cmap='gray')




