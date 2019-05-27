## @package utils
#  Basic functions used in the project.
# @author Thiago Pincinato and Tamara Melle

import matplotlib.pyplot as plt
import numpy as np

## It converts a rbg image to a gray scale image with value between 0 and 25500.
    #  @param img_ image to be converted.
    #  @return A gray scale image with value between 0 and 25500.


def rgb_2_gray(img_):
    out_ = np.multiply(img_[:, :, 0], 0.3) + np.multiply(img_[:, :, 1], 0.59) + np.multiply(img_[:, :, 2], 0.11)
    out_ = np.multiply(out_, 25500).astype('uint16')
    return out_


## It generates subplots.
    #  @param figure_number number of the figure.
    #  @param suplot_number number of the subplot.
    #  @param plot_1 image to be plotted first.
    #  @param plot_2 second image to be plotted.
    #  @param title1 title of first subplot.
    #  @param title2 title of second subplot.
    #  @param op1 parameters to select if first image is gray scale or not.
    #  0 if not gray scale and 1 if it is.
    #  @param op1 parameters to select if second image is gray scale or not.
    #  0 if not gray scale and 1 if it is.


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




