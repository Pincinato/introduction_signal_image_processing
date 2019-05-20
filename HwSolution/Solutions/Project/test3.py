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


filename = 'input_8_1'


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


def find_new_bottom(t_point, img_):
    main_value = img_[t_point, 0]
    deviation = main_value/4
    points_to_analyse= np.where(img_ > (main_value - deviation))
    new_bottom_ = points_to_analyse[0][len(points_to_analyse[1]) - 1]
    return new_bottom_


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


def k_means_clustering(img_gray_, mask):
    img_to_work = img_gray_ * mask
    img_to_work = scipy.ndimage.gaussian_filter(img_to_work, sigma=3)
    #z = img.reshape((-1, 1)
    z = img_to_work
    z = np.float32(z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 64
    ret, label, center = cv2.kmeans(z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(img_gray_.shape)
    plt.figure(5)
    plt.title('Result of K means clustering')
    plt.imshow(res2, cmap='gray')


def adaptive_threshold(img_gray_, mask):
    img_to_work = img_gray_ * mask
    img_threshold = cv2.adaptiveThreshold(img_to_work, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    plt.figure(5)
    plt.title('Result of threshold')
    plt.imshow(img_threshold, cmap='gray')


def find_grad_Iy(img_gray_):
    filtered_img = scipy.ndimage.gaussian_filter(img_gray_, sigma=3)
    img_padded = np.pad(filtered_img, [(1, 1), (1,1)], 'constant', constant_values=(0, 0))
    # getting gradients
    Iy = scipy.signal.convolve(img_padded, np.array([[-1, 0, 1]]).T, mode='same', method='auto')
    # Ix = scipy.signal.convolve(img_padded, np.array([[-1, 0, 1]]), mode='same', method='auto')
    # removing negatives values
    Iyy = np.power(Iy, 2)
    # Iyy = np.sqrt(np.power(Iy, 2) + np.power(Ix, 2))
    # normalizing and generating 0-255 range values
    Iy_normalized = np.array(np.multiply(np.divide(Iyy, np.amax(Iyy)), 255), dtype=np.uint8)
    plt.figure(14)
    plt.title('Result of Y gradient')
    plt.imshow(Iyy, cmap='gray')
    return Iy_normalized[1:Iy_normalized.shape[0]-1, 1:Iy_normalized.shape[1]-1]


def find_grad_Iy_masked(img_gray_, mask):
    img_to_work = img_gray_ * mask
    filtered_img = scipy.ndimage.gaussian_filter(img_to_work, sigma=3)
    img_padded = np.pad(filtered_img, [(1, 1)], 'constant', constant_values=(0, 0))
    # getting gradients
    Iy = scipy.signal.convolve(img_padded, np.array([[-1, 0, 1]]).T, mode='same', method='auto')
    # removing negatives values
    Iyy = np.power(Iy, 2)
    # normalizing and generating 0-255 range values
    Iy_normalized = np.array(np.multiply(np.divide(Iyy, np.amax(Iyy)), 255), dtype=np.uint8)
    plt.figure(6)
    plt.title('Result of Y gradient after mask')
    plt.imshow(Iy_normalized, cmap='gray')
    return Iy_normalized[1:Iy_normalized.shape[0]-1, 1:Iy_normalized.shape[1]-1]


def threshold_histogram(img_gray_, mask):
    img_to_work = np.multiply(img_gray_, mask)
    img_normalized = np.multiply(np.divide(img_to_work, np.amax(img_to_work)), 255)
    img_histogram = np.histogram(img_normalized, bins=10)
    plt.figure(3)
    plt.subplot(1, 2, 1)
    plt.title('Result of histogram')
    plt.hist(img_histogram, bins=10, range=(0, 255), density='True')
    threshold_mask = np.where(img_normalized >= img_histogram[1][8])
    newImage = np.zeros(img_to_work.shape)
    newImage[threshold_mask] = img_to_work[threshold_mask]
    plt.subplot(1, 2, 2)
    plt.title('Result of threshold')
    plt.imshow(newImage, cmap='gray')
    return threshold_mask



def selective_search(img_, mask):
    img_to_work = np.multiply(img_, mask).astype('float32')
    im = cv2.cvtColor(img_to_work, cv2.COLOR_GRAY2BGR)
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)
    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    # setting imageimg_to_work
    ss.setBaseImage(im)
    # run selective search segmentation on input image
    ss.switchToSelectiveSearchFast()
    # ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    numShowRects = 10
    for i, rect in enumerate(rects):
        # draw rectangle for region proposal till numShowRects
        if (i < numShowRects):
            x, y, w, h = rect
            cv2.rectangle(img_to_work, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
        else:
            break
    plt.figure(5)
    plt.title('Result of selective_search')
    plt.imshow(img_to_work, cmap='gray')


def graph_search(img_, mask):
    img_to_work = np.multiply(img_, mask).astype('float32')
    img_to_work = img_
    im = cv2.cvtColor(img_to_work, cv2.COLOR_GRAY2BGR)
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)
    ss = cv2.ximgproc.segmentation.createGraphSegmentation()
    # ss.setK(10)
    # ss.setSigma(5)
    # ss.setMinSize(5)
    print(ss.getK(), ss.getSigma(), ss.getMinSize())
    outImage = np.copy(im)
    ss.processImage(im, outImage)
    plt.figure(5)
    plt.title('Result of graph_search')
    plt.imshow(outImage, cmap='gray')


def test(img_, mask, ilm_mask,t,b):
    #img_to_work = feature.canny(img_, 3)
    #outImage = np.multiply(img_, mask & ilm_mask)
    #outImage = np.multiply(outImage, img_).astype('uint8')
    outImage = find_grad_Iy_masked(img_, mask & ilm_mask)
    plt.figure(10)
    plt.subplot(1, 2, 1)
    plt.title('Received image')
    plt.imshow(img_, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Result of test')
    plt.imshow(outImage, cmap='gray')
    return 0


def remove_ilm(img_, mask):
    mask = mask | np.roll(mask, 4, axis=0) | np.roll(my_mask_ilm, 8, axis=0)
    mask = -1*mask + 1
    img_to_work = np.multiply(img_, mask)
    plt.figure(7)
    plt.title('Result of ilm removal')
    plt.imshow(img_to_work, cmap='gray')
    plt.plot(mask)
    return 0


def plot_mask_result(img_, mask):
    img_to_work = np.multiply(img_, mask)
    plt.figure(9)
    plt.title('Result of mask')
    plt.imshow(img_to_work, cmap='gray')
    return 0

img = imread(filename + '.png')
if len(img.shape) == 3:
    img_gray = img[:, :, 0]
else:
    img_gray = img

(top, bottom) = find_top_bottom(img_gray)
my_mask = using_grab_cut(top, bottom, img_gray)
my_grad_y = find_grad_Iy_masked(img_gray,my_mask)
(new_top, new_bottom) = find_new_top_bottom(my_grad_y)
my_mask_ilm = using_grab_cut(new_top, new_bottom + 10, my_grad_y)
my_mask_ilm = my_mask_ilm | np.roll(my_mask_ilm, 4, axis=0) | np.roll(my_mask_ilm, 8, axis=0)
my_mask_ilm = -1*my_mask_ilm + 1
remove_ilm(img_gray, my_mask_ilm)

my_Iy = find_grad_Iy_masked(img_gray, my_mask_ilm)
#(top, bottom) = find_top_bottom(my_Iy)
# using_grab_cut(top,bottom,img_gray)
plt.figure(10)
plt.imshow(my_Iy, cmap='gray')
plt.show()
