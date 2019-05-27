## @package MyTester
#  My Tester aims to test the principal functions of the project.
#  Those test consist of routine calls, time measurement and parameters deviations
# @author Thiago Pincinato and Tamara Melle

import numpy as np
import matplotlib.pyplot as plt
import time
import mainSegmentation as sg
import DetectionSNF as dt
import utils
import cv2

## It tests the function using_grab_cut for a array of images.
# In additional, it measures the time needed to execute such a function.
    #  @param img_array Array containing all images to be tested.


def test_grab_cut(img_array):
    j = 1
    for i in np.arange(img_array.__len__()):
        print("image "+str(i+1)+"/"+str(img_array.__len__()))
        start_time = time.time()
        if len(img_array[i].shape) == 3:
            img_gray = img_array[i][:, :, 0]
        else:
            img_gray = img_array[i]
        plt.figure(i+j)
        j = j+1
        (top, bottom, left, right) = sg.find_top_bottom_left_right(img_gray)
        mask = sg.using_grab_cut(top, bottom, left, right, img_gray)
        cropped_ = sg.crop_image(img_gray, mask)
        plt.figure(i+j)
        plt.imshow(cropped_,cmap='gray')
        j = j+1
        elapsed_time = time.time() - start_time
        print("Execution time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(int(elapsed_time % 60)) + " seconds")
        print("image "+str(i+1)+"/"+str(img_array.__len__())+" done")

## It tests the segmentation pathway for a array of images, using several combinations of the parameters k, min and sigma .
# In additional, it measures the time needed to execute such a function.
    #  @param img_array Array containing all images to be tested.


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


## It tests the segmentation pathway.
# In additional, it measures the time needed to execute such a function.
    #  @param img_array Array containing all images to be tested.
    #  @param sigma sigma value required by the sg.segment_image function
    #  @param k k value required by the sg.segment_image function
    #  @param min_ min_ value required by the sg.segment_image function


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


## It tests the detection pathway.
# In additional, it measures the time needed to execute such a function.
    #  @param img_array Array containing all images to be tested.
    #  @param sigma sigma value required by the sg.segment_image function
    #  @param k k value required by the sg.segment_image function
    #  @param min_ min_ value required by the sg.segment_image function


def test_detection(img_array, sigma, k, min_):
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
        elapsed_time = time.time() - start_time
        print("Segmentation time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(int(elapsed_time % 60)) + " seconds")
        start_time = time.time()
        detection_result.append(dt.detect_SNF(grabcut_image, cropped_image, grabcut_mask))
        elapsed_time = time.time() - start_time
        print("Classification time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(int(elapsed_time % 60)) + " seconds")
        print("image "+str(i+1)+"/"+str(img_array.__len__())+" done")
    return detection_result


## It tests the detection pathway, presuming that all images have a SNF.
    #  @param img_array Array containing all images to be tested.
    #  @param sigma sigma value required by the sg.segment_image function
    #  @param k k value required by the sg.segment_image function
    #  @param min_ min_ value required by the sg.segment_image function
    #  @return Ratio of positive SNF detection over the length of img_array

def test_detection_positive_rate(img_array_, sigma, k, min_):
    images_with_SRF = test_detection(img_array_, sigma, k , min_)
    return np.sum(images_with_SRF) / (img_array_.__len__())


## It tests the detection pathway, presuming that all images do not have a SNF.
    #  @param img_array Array containing all images to be tested.
    #  @param sigma sigma value required by the sg.segment_image function
    #  @param k k value required by the sg.segment_image function
    #  @param min_ min_ value required by the sg.segment_image function
    #  @return Ratio of negative SNF detection over the length of img_array


def test_detection_negative_rate(img_array_, sigma, k , min_):
    images_with_NoSRF = test_detection(img_array_, sigma, k , min_)
    return ((img_array_.__len__()) - np.sum(images_with_NoSRF)) / (img_array_.__len__())


## It tests the detection pathway, plot the results or save the result in the Output folder.
# In additional, it measures the time needed to execute such a function.
    #  @param img_array Array containing all images to be tested.
    #  @param sigma sigma value required by the sg.segment_image function
    #  @param k k value required by the sg.segment_image function
    #  @param min_ min_ value required by the sg.segment_image function
    #  @param save_or_plot Parameter to select if the output will be saved in a folder or plotted
#  0 - It saves output in a folder
#  1 - It plots the output


def test_detection_SNF(img_array, sigma, k, min_, save_or_plot=0):
    i = 0
    for actual_image in img_array:
        print("image " + str(i + 1) + "/" + str(img_array.__len__()))
        start_time = time.time()
        if len(actual_image.shape) == 3:
            img_gray = actual_image[:, :, 0]
        else:
            img_gray = actual_image
        # Segmentation part -> to have plots , debug = 1
        (seg_image, cropped_image, grabcut_mask) = sg.segment_image(img_gray, sigma, k, min_)
        elapsed_time = time.time() - start_time
        print("Segmentation time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(int(elapsed_time % 60)) + " seconds")
        start_time = time.time()
        mask = dt.find_SNF_mask(seg_image, cropped_image, grabcut_mask)
        new_mask = np.zeros([cropped_image.shape[0], cropped_image.shape[1]])
        for ind in np.arange(mask.__len__()):
            new_mask = new_mask + mask[ind]
        new_mask[np.where(new_mask > 0)] = 255
        if save_or_plot == 1:
            plt.figure(i)
            plt.subplot(3, 2, 1)
            plt.title('Original Image')
            plt.imshow(img_gray, cmap='gray')
            plt.subplot(3, 2, 2)
            plt.title('Cropped image (GraphCut)')
            plt.imshow(cropped_image, cmap='gray')
            plt.subplot(3, 2, 3)
            plt.title('Segmentation')
            plt.imshow(seg_image)
            plt.subplot(3, 2, 4)
            plt.title('SNF detection')
            plt.imshow(new_mask, cmap='gray')
            plt.subplot(3, 2, 6)
            plt.title('Final  result')
            plt.imshow(cropped_image, cmap='gray')
            plt.imshow(new_mask, alpha=0.5)
        if save_or_plot == 0:
            output_img = np.zeros([cropped_image.shape[0], cropped_image.shape[1],3])
            output_img[:, :, 0] = cropped_image
            output_img[:, :, 1] = cropped_image
            output_img[:, :, 2] = new_mask
            cv2.imwrite("OutputImages/output_image_new-" + str(i) + ".jpg",output_img)
        elapsed_time = time.time() - start_time
        print("Classification time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(int(elapsed_time % 60)) + " seconds")
        print("image "+str(i+1)+"/"+str(img_array.__len__())+" done")
        i = i+1
