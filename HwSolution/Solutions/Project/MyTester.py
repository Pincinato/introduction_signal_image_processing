import numpy as np
import matplotlib.pyplot as plt
import time
import mainSegmentation as sg
import DetectionSNF as dt
import utils
from skimage import feature
import cv2

# testing grab cut
def test_grab_cut(img_array):
    for i in np.arange(img_array.__len__()):
        print("image "+str(i+1)+"/"+str(img_array.__len__()))
        start_time = time.time()
        if len(img_array[i].shape) == 3:
            img_gray = img_array[i][:, :, 0]
        else:
            img_gray = img_array[i]
        (top, bottom, left, right) = sg.find_top_bottom_left_right(img_gray)
        sg.using_grab_cut(top, bottom, left, right, img_gray)
        elapsed_time = time.time() - start_time
        print("Execution time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(int(elapsed_time % 60)) + " seconds")
        print("image "+str(i+1)+"/"+str(img_array.__len__())+" done")


def test_seg_k_m_sigma(img_array):
    n_figure = 0
    for i in np.arange(img_array.__len__()):
        print("image " + str(i + 1) + "/" + str(img_array.__len__()))
        start_time = time.time()
        if len(img_array[i].shape) == 3:
            img_gray = img_array[i][:, :, 0]
        else:
            img_gray = img_array[i]
        # Segmentation part -> to have plots , debug = 1
        for k_ in [10.0, 50.0, 90.0, 130,  170,  250,  300, 400, 500]:
            start_time = time.time()
            n_figure = n_figure + 1
            j = 1
            for min_ in [10.0, 30,  90, 150, 200 , 320, 410, 520, 650]:
                plt.figure(n_figure)
                for sigma in [3]:
                    (out_img, cropped_image, mask__) = sg.segment_image(img_gray, sigma, k_, min_)
                    plt.subplot(5, 2, j)
                    plt.title('k_' +str(k_)+' min_' +str(min_)+' sigma_' +str(sigma))
                    plt.imshow(out_img)
                    j = j+1
            elapsed_time = time.time() - start_time
            print("Execution time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(
            int(elapsed_time % 60)) + " seconds")
            print("image " + str(i + 1) + "/" + str(img_array.__len__()) + " k_"+str(k_) )
        print("image " + str(i + 1) + "/" + str(img_array.__len__()) + " done")




def test_segmentation(img_array, sigma, k, min_):
    j = 1
    n_figure = 0
    for i in np.arange(img_array.__len__()):
        print("image "+str(i+1)+"/"+str(img_array.__len__()))
        start_time = time.time()
        if len(img_array[i].shape) == 3:
            img_gray = img_array[i][:, :, 0]
        else:
            img_gray = img_array[i]
        # Segmentation part -> to have plots , debug = 1
        (out_img, cropped_image, mask__) = sg.segment_image(img_gray, sigma, k, min_)
        # ploting
        if (i > 3) and (i < 5):
            n_figure = 1
            j = 1
        if (i > 7) and (i < 9):
            n_figure = 2
            j = 1
        if (i > 11) and (i < 13):
            n_figure = 3
            j = 1
        utils.subplot(n_figure + 1, j, cropped_image,out_img, 'Cropped Image', 'Result of Graph based segmentation in RGB', 1, 0)
        elapsed_time = time.time() - start_time
        print("Execution time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(int(elapsed_time % 60)) + " seconds")
        print("image "+str(i+1)+"/"+str(img_array.__len__())+" done")
        j = j + 2


def test_detection(img_array, sigma, k , min_, show=1):
    detection_result =[]
    for i in np.arange(img_array.__len__()):
        print("image " + str(i + 1) + "/" + str(img_array.__len__()))
        start_time = time.time()
        if len(img_array[i].shape) == 3:
            img_gray = img_array[i][:, :, 0]
        else:
            img_gray = img_array[i]
        # Segmentation part -> to have plots , debug = 1
        (grabcut_image, cropped_image, grabcut_mask) = sg.segment_image(img_gray, sigma, k, min_)
        if show == 1 :
            plt.figure(i + 19)
            plt.imshow(grabcut_image)
            plt.figure(i+20)
            plt.imshow(cropped_image)
        elapsed_time = time.time() - start_time
        print("Segmentation time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(int(elapsed_time % 60)) + " seconds")
        start_time = time.time()
        if show == 1:
            plt.figure(i + 21)
        detection_result.append(dt.detect_SNF(grabcut_image, cropped_image, grabcut_mask))
        elapsed_time = time.time() - start_time
        print("Classification time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(int(elapsed_time % 60)) + " seconds")
        print("image "+str(i+1)+"/"+str(img_array.__len__())+" done")
        if show == 1:
            plt.show()
    return detection_result


def test_detection_positive_rate(img_array_, sigma, k , min_):
    images_with_SRF = test_detection(img_array_, sigma, k , min_, show=0)
    return np.sum(images_with_SRF) / (img_array_.__len__())


def test_detection_negative_rate(img_array_, sigma, k , min_):
    images_with_SRF = test_detection(img_array_, sigma, k , min_, show=0)
    return ((img_array_.__len__()) - np.sum(images_with_SRF)) / (img_array_.__len__())
