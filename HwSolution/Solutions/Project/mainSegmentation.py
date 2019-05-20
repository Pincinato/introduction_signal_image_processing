import matplotlib.pyplot as plt
import numpy as np
from skimage import feature
from scipy.ndimage import gaussian_filter
import cv2
import matplotlib.patches as patches
import GraphBasedTest as gb
import scipy

DEBUG = 1


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


def find_new_top_bottom(img_gray):
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
    if Bottom_point > Top_point + 70:
        Bottom_point = Top_point + 70
    return Top_point, Bottom_point,


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


def crop_image(img_, mask):
    cropped_image_ = np.multiply(img_, mask)
    points_to_analyse = np.where(cropped_image_ > 0)
    Top_point_ = points_to_analyse[0][1]
    Bottom_point_  = points_to_analyse[0][len(points_to_analyse[1]) - 1]
    Left_point = np.amin(points_to_analyse[1])
    Right_point = np.amax(points_to_analyse[1])
    cropped_image_ = cropped_image_[Top_point_:Bottom_point_, Left_point:Right_point]
    return cropped_image_


def enhance_contrast(img_gray_):
    out_img_ = cv2.equalizeHist(img_gray_)
    #out_img_[np.where(out_img_ > 190)] = 255
    #out_img_[np.where(out_img_ < 50)] = 0
    return out_img_


def segment_image(img_gray, sigma=3, k=200, min_=50, debug=0):

    global DEBUG
    DEBUG = debug
    (top, bottom, left, right) = find_top_bottom_left_right(img_gray)
    my_mask = using_grab_cut(top, bottom, left, right, img_gray)
    cropped_image = crop_image(img_gray, my_mask)
    cropped_image = enhance_contrast(cropped_image)
    cropped_image_rgb = np.zeros([cropped_image.shape[0], cropped_image.shape[1], 3])
    cropped_image_rgb[:, :, 0] = np.copy(cropped_image)
    cropped_image_rgb[:, :, 1] = np.copy(cropped_image)
    cropped_image_rgb[:, :, 2] = np.copy(cropped_image)
    out = gb.segment(cropped_image_rgb, sigma, k, min_)
    out = np.divide(out, np.amax(out))
    cropped_mask = crop_image(my_mask, my_mask)
    return out, cropped_image, cropped_mask
