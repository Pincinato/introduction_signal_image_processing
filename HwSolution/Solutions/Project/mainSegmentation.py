## @package mainSegmentation
#  This py file is responsible to remove the background noise, to crop the image in the region of interest
#  and to segment the cropped image using a graph based segmentation algorithm.
# @author Thiago Pincinato and Tamara Melle

import matplotlib.pyplot as plt
import numpy as np
from skimage import feature
import cv2
import matplotlib.patches as patches
import GraphBasedTest as gb

DEBUG = 1

## It finds the margins of the image, based on the canny edge detection.
    #  @param img_gray original image.
    #  @return (Top_point, Bottom_point, Left_point, Right_pointTrue).


def find_top_bottom_left_right(img_gray):
    # original image
    if DEBUG:
        plt.subplot(2, 2, 1)
        plt.title('Original Image')
        plt.imshow(img_gray, cmap='gray')
    # Canny edge
    edges = feature.canny(img_gray, 3)
    if DEBUG:
        plt.subplot(2, 2, 2)
        plt.title('Canny edge detection')
        plt.imshow(edges, cmap='gray')
    # pdb.set_trace()
    points_to_analyse = np.where(edges == True)
    Top_point = points_to_analyse[0][1]
    Bottom_point = points_to_analyse[0][len(points_to_analyse[1]) - 1]
    Left_point = np.amin(points_to_analyse[1])
    Right_point = np.amax(points_to_analyse[1])
    if Bottom_point - Top_point > 300:
        Top_point = Bottom_point - 300
    return Top_point, Bottom_point, Left_point, Right_point


## It finds the top and bottom margins of the image, based on the canny edge detection.
    #  @param img_gray original image.
    #  @return (Top_point, Bottom_point).


def find_new_top_bottom(img_gray):
    # original image
    if DEBUG:
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(img_gray, cmap='gray')
    # Canny edge
    edges = feature.canny(img_gray, 3)
    if DEBUG:
        plt.subplot(1, 2, 2)
        plt.title('Canny edge detection')
        plt.imshow(edges, cmap='gray')
    # pdb.set_trace()
    points_to_analyse = np.where(edges == True)
    Top_point = points_to_analyse[0][1]
    Bottom_point = points_to_analyse[0][len(points_to_analyse[1]) - 1]
    if Bottom_point > Top_point + 70:
        Bottom_point = Top_point + 70
    return Top_point, Bottom_point,

## It removes the background noise .
    #  @param top_point the most top point where below all foreground pixels can be found.
    #  @param bottom_point the most bottom point where above all foreground pixels can be found.
    #  @param left_point the most left point where on the right all foreground pixels can be found.
    #  @param right_point the most right point where on the left all foreground pixels can be found.
    #  @param img_gray original image.
    #  @return image without background noise.

def using_grab_cut(top_point, bottom_point, left_point, right_point, img_gray):
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)
    # reading image
    test = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    # creating mask background, foreground models
    mask = np.zeros(test.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    # creating rect
    rect = (left_point, top_point, right_point - left_point, bottom_point - top_point + 10)
    # applying grabCut
    cv2.grabCut(test, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
    # creating mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # applying mask
    imgToPlot = test * mask2[:, :, np.newaxis]
    # plots rect
    im = img_gray
    rectangle = patches.Rectangle((left_point, top_point), right_point - left_point, bottom_point - top_point + 10, linewidth=1, edgecolor='r', facecolor='none')
    if DEBUG:
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


## It make crop on an image, based on a mask .
    #  @param img_gray original image.
    #  @param mask used to make a crop.
    #  @return cropped image.


def crop_image(img_, mask):
    cropped_image_ = np.multiply(img_, mask)
    points_to_analyse = np.where(cropped_image_ > 0)
    Top_point_ = points_to_analyse[0][1]
    Bottom_point_  = points_to_analyse[0][len(points_to_analyse[1]) - 1]
    Left_point = np.amin(points_to_analyse[1])
    Right_point = np.amax(points_to_analyse[1])
    cropped_image_ = cropped_image_[Top_point_:Bottom_point_, Left_point:Right_point]
    return cropped_image_


## It enhance the contrast.
    #  @param img_gray original image.
    #  @return image with better contrast.


def enhance_contrast(img_gray_):
    out_img_ = cv2.equalizeHist(img_gray_)
    return out_img_

## It segment an image, based on graph-based approach.
    #  @param img_gray original image.
    #  @param sigma sigma used in the segment algorithm (determines the amount smooth).
    #  @param k k used in the segment algorithm (determines the how big the regions will be).
    #  @param min_ min_ used in the segment algorithm (determines the minimum criteria to merge to graphs).
    #  @param debug parameters determines if there will be plots of the segmentation process or not.
    #  0 - no plot. 1- plots
    #  @return out(segmented image) , cropped_image(original image without background and cropped) ,
    #  cropped_mask ( mask used to crop image)

def segment_image(img_gray, sigma=3, k=200, min_=50, debug=0):

    global DEBUG
    DEBUG = debug
    (top, bottom, left, right) = find_top_bottom_left_right(img_gray)
    my_mask = using_grab_cut(top, bottom, left, right, img_gray)
    if np.sum(my_mask > 0) == 0:
        cropped_image = img_gray
    else:
        cropped_image = crop_image(img_gray, my_mask)
    cropped_image = enhance_contrast(cropped_image)
    cropped_image_rgb = np.zeros([cropped_image.shape[0], cropped_image.shape[1], 3])
    cropped_image_rgb[:, :, 0] = np.copy(cropped_image)
    cropped_image_rgb[:, :, 1] = np.copy(cropped_image)
    cropped_image_rgb[:, :, 2] = np.copy(cropped_image)
    out = gb.segment(cropped_image_rgb, sigma, k, min_)
    out = np.divide(out, np.amax(out))
    if np.sum(my_mask > 0) == 0:
        cropped_mask = np.ones(my_mask.shape)
    else:
        cropped_mask = crop_image(my_mask, my_mask)
    return out, cropped_image, cropped_mask
