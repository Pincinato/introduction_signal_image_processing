from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import random
import os.path
from scipy.misc import imread
from scipy import sparse
import scipy
import utils as utls
import pdb
from scipy import signal
from skimage import feature
import cv2
from PIL import Image
import matplotlib.patches as patches
import GraphBasedTest as gb


filename = 'input_9_1'


def find_top_bottom(img_gray):
    # original image
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(img_gray, cmap='gray')
    # Canny edge
    edges = feature.canny(img_gray, 3)
    plt.subplot(2, 2, 2)
    plt.title('Canny edge detection')
    plt.imshow(edges, cmap='gray')
    # pdb.set_trace()
    points_to_analyse = np.where(edges == True)
    Top_point = points_to_analyse[0][1]
    Bottom_point = points_to_analyse[0][len(points_to_analyse[1]) - 1]
    return Top_point, Bottom_point


def find_new_top_bottom(img_gray):
    # original image
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(img_gray, cmap='gray')
    # Canny edge
    edges = feature.canny(img_gray, 3)
    plt.subplot(2, 2, 2)
    plt.title('Canny edge detection')
    plt.imshow(edges, cmap='gray')
    # pdb.set_trace()
    points_to_analyse = np.where(edges == True)
    Top_point = points_to_analyse[0][1]
    Bottom_point = points_to_analyse[0][len(points_to_analyse[1]) - 1]
    if Bottom_point > Top_point + 70:
        Bottom_point = Top_point + 70
    return Top_point, Bottom_point


def using_grab_cut(top_point, bottom_point, img_gray):
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)
    # reading image
    test = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    # creating mask background, foreground models
    mask = np.zeros(test.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    # creating rect
    rect = (0, top_point - 5, test.shape[1] - 3, bottom_point - top_point + 10)
    # applying grabCut
    cv2.grabCut(test, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    # creating mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # applying mask
    imgToPlot = test * mask2[:, :, np.newaxis]
    # plots rect
    im = img_gray
    rectangle = patches.Rectangle((0, top_point-5), test.shape[1] - 3, bottom_point - top_point + 10, linewidth=1, edgecolor='r', facecolor='none')
    plt.subplot(2, 2, 3)
    plt.imshow(im, cmap='gray')
    ax = plt.gca()
    ax.add_patch(rectangle)
    plt.title('Rectangle')
    # ploting result
    plt.subplot(2, 2, 4)
    plt.title('Result of GrabCut')
    plt.imshow(imgToPlot, cmap='gray')
    return mask2[:, :, np.newaxis][:, :, 0]


def crop_image(img_, mask):
    cropped_image_ = np.multiply(img_, mask)
    points_to_analyse = np.where(cropped_image_ > 0)
    Top_point_ = points_to_analyse[0][1]
    Bottom_point_  = points_to_analyse[0][len(points_to_analyse[1]) - 1]
    cropped_image_ = cropped_image_[Top_point_:Bottom_point_, :]
    return cropped_image_


def extract_mask_middle_layers(img_gray_):
    img_to_work_ = np.multiply(img_gray_,255).astype('uint8')
    hist = np.histogram(img_to_work_, bins=np.arange(256))
    middle_layer_color = hist[1][np.argmax(hist[0])]
    found_mask = (img_to_work_ == middle_layer_color)
    #pdb.set_trace()
    return found_mask

img = imread(filename + '.png')
if len(img.shape) == 3:
    img_gray = img[:, :, 0]
else:
    img_gray = img

(top, bottom) = find_top_bottom(img_gray)
my_mask = using_grab_cut(top, bottom, img_gray)
cropped_image = crop_image(img_gray, my_mask)
sigma = 3
k = 200 #500 # 200
min = 50  # 50
cropped_image_rgb = np.zeros([cropped_image.shape[0], cropped_image.shape[1], 3])
cropped_image_rgb[:, :, 0] = np.copy(cropped_image)
cropped_image_rgb[:, :, 1] = np.copy(cropped_image)
cropped_image_rgb[:, :, 2] = np.copy(cropped_image)
out = gb.segment(cropped_image_rgb, sigma, k, min)
out = np.divide(out, np.amax(out))
plt.figure(9)
plt.subplot(1,2,1)
plt.title('Result of Graph based segmentation in RGB')
plt.imshow(out)
out_gray = np.multiply(out[:, :, 0], 0.3) + np.multiply(out[:, :, 1], 0.59) + np.multiply(out[:, :, 2], 0.11)
plt.subplot(1,2,2)
plt.title('Result of Graph based segmentation in gray')
plt.imshow(out_gray,cmap='gray')

mask = extract_mask_middle_layers(out_gray)
plt.figure(77)
plt.subplot(1,2,1)
plt.title('Result middle layer mask')
plt.imshow(mask,cmap='gray')

plt.subplot(1,2,2)
plt.title('Appling mask to cropped_image')
plt.imshow(np.multiply(cropped_image,mask),cmap='gray')

plt.show()