## @package DetectionSNF
#  Package developed for detecting SNF in a graph-based segmented OCT image.
# @author Thiago Pincinato and Tamara Melle

import numpy as np
import utils

## It computes the ratio of black pixel over the amount of pixel in an image.
    #  @param section_to_analyse image in which the black pixels will be counted.
    #  @param threshold_ threshold used to determine what is a black pixel.
    #  @return The ratio of black pixel over the amount of pixel in an image.

def black_pixel_density(section_to_analyse, threshold_):
    amount_of_black = np.sum(section_to_analyse < threshold_)
    amount_of_pixels = np.sum(section_to_analyse < 256)
    proportion_black_pixel = amount_of_black / amount_of_pixels
    return proportion_black_pixel


## It counts the amount of pixel in an images.
    #  @param section image in which pixels bigger than 0 will be counted.
    #  @return amount of pixels bigger than 0.

def count_pixels(section):
    return np.sum((section > 0))


## It compute the histogram of an image.
    #  @param gray_img_ image used to obtain the histogram.
    #  @return histogram.


def get_hist(gray_img_):
    (hist_, ed) = np.histogram(gray_img_, bins=np.arange(25500))
    return np.argwhere(hist_ != 0), hist_[np.where(hist_ != 0)]


## It remove large regions in a histogram (remove big values).
    #  @param hist_ histogram.
    #  @param total_pixels total pixel in the image.
    #  @param threshold_ percentage to determine large area, based on the total pixels.
    #  @return new histogram without large regions.


def remove_large_regions(hist_, total_pixels, threshold_):
    criteria = total_pixels*threshold_
    indexes = np.where(hist_[1] < criteria)
    ret = hist_[0][indexes], hist_[1][indexes]
    return ret


## It generates masks/candidates (extracted from a histogram) in a single ndarray.
    #  @param img_gray_ image used to localize the position of each pixel in a mask.
    #  @param hist_ histogram used to localize the position of each pixel in a mask.
    #  @return a ndarray with all masks/candidates.

def single_mask_from_histrogram(img_gray_, hist_):
    mask_ = np.zeros(img_gray_.shape)
    for i in np.arange(hist_[1].size):
        indexes = np.where(img_gray_ == hist_[0][i])
        mask_[indexes] = 1
    return mask_


## It generates masks/candidates extracted from a histogram.
    #  @param img_gray_ image used to localize the position of each pixel in a mask.
    #  @param hist_ histogram used to localize the position of each pixel in a mask.
    #  @return an object with arrays of each mask/candidate.


def creating_masks_from_histogram(img_gray_, hist_):
    masks_ = []
    for i in np.arange(hist_[1].size):
        masks_.append( np.zeros(img_gray_.shape))
        indexes = np.where(img_gray_ == hist_[0][i])
        masks_[i][indexes] = 1
    return masks_


## It removes candidates with small amout of black pixels.
    #  @param masks_ candidates.
    #  @param img_ original image .
    #  @param threshold minimum black pixel density required for a certain region.
    #  @param threshold_black_pixel_ threshold to determine what is a black pixel.
    #  @return a new mask with the remaining candidates.


def remove_low_density_black_pixel(masks_, img_, threshold, threshold_black_pixel_):
    new_masks = []
    for actual_mask in masks_:
        temp_mask = np.zeros(actual_mask.shape)
        # masking by add 256 which certainly remove area outside the mask from the black pixel counting
        temp_mask[np.where(actual_mask == 0)] = 256
        section = img_ + temp_mask
        density = black_pixel_density(section, threshold_black_pixel_)
        if density > threshold:
            new_masks.append(actual_mask)
    return new_masks


## It computes the density of pixel in a mask.
    #  @param actual_masks candidates.
    #  @return the ratio of total pixels bigger than zero, over all pixels.


def compute_pixel_density(actual_mask):
    points_to_analyse = np.where((actual_mask > 0) == True)
    top_ = points_to_analyse[0][0]
    bottom_ = points_to_analyse[0][len(points_to_analyse[1]) - 1]
    left_ = np.amin(points_to_analyse[1])
    right_ = np.amax(points_to_analyse[1])
    total_region = (bottom_ - top_)*(right_ - left_)
    total_pixels = np.sum(actual_mask == 1)
    return total_pixels/total_region


## It removes candidates with low density of pixels.
    #  @param masks_ candidates.
    #  @param threshold minimum percentage to be classified as a low density mask.
    #  @return a new mask with the remaining candidates.


def remove_low_density_candidates(masks_,threshold_):
    new_masks = []
    for actual_mask in masks_:
        density = compute_pixel_density(actual_mask)
        if density > threshold_:
            new_masks.append(actual_mask)
    return new_masks

## It removes the candidates that are in the border.
    #  @param masks_ candidates.
    #  @param img_ original image.
    #  @return a new mask with the remaining candidates.


def remove_in_border(masks_, img_):
    new_masks = []
    for actual_mask in masks_:
        m_points_to_analyse = np.where((actual_mask > 0) == True)
        img_points_to_analyse = np.where((img_ > 0) == True)
        m_top_ = m_points_to_analyse[0][0]
        m_bottom_ = m_points_to_analyse[0][len(m_points_to_analyse[1]) - 1]
        m_left_ = np.amin(m_points_to_analyse[1])
        m_right_ = np.amax(m_points_to_analyse[1])
        img_top_ = img_points_to_analyse[0][0]
        img_bottom_ = img_points_to_analyse[0][len(img_points_to_analyse[1]) - 1]
        img_left_ = np.amin(img_points_to_analyse[1])
        img_right_ = np.amax(img_points_to_analyse[1])
        if (m_top_ > img_top_ + 5) and (m_bottom_ < img_bottom_ - 5):
            if (m_left_ > img_left_ + 5) and (m_right_ < img_right_ - 5):
                new_masks.append(actual_mask)
    return new_masks


## It detects if candidate is in/above the ilm layer.
# In additional, it measures the time needed to execute such a function.
    #  @param img_ candidate to be tested.
    #  @return True if the candidate is above the ilm layer. False otherwise.


def detect_possible_ilm(img_):
    count = np.sum(img_ == 0)
    area = img_.shape[1]
    ack = False
    if count >= area:
        ack = True
    return ack


## It removes the candidates that are in or above the ilm layer.
    #  @param masks_ candidates.
    #  @param img_ original image.
    #  @param depth_to_use number of lines to be analysed.
    #  @return a new mask with the remaining candidates.

def remove_above_ilm(masks_, img_, depth_to_use=3):
    new_masks = []
    for actual_mask in masks_:
        m_points_to_analyse = np.where((actual_mask > 0) == True)
        m_top_ = m_points_to_analyse[0][0]
        m_left_ = np.amin(m_points_to_analyse[1])
        m_right_ = np.amax(m_points_to_analyse[1])
        if (m_top_ - depth_to_use > 0):
            if detect_possible_ilm(img_[m_top_ - depth_to_use :m_top_, m_left_ :m_right_]) == False:
                new_masks.append(actual_mask)
    return new_masks


## It detects if candidate is in/above the rpe layer.
# In additional, it measures the time needed to execute such a function.
    #  @param img_ original image.
    #  @param mask_ candidate to be tested.
    #  @param thre_white threshold to determine what is a white pixel.
    #  @return True if the candidate is above the rpe layer. False otherwise.


def detect_possible_rpe(img_,mask_, thre_white=180):
    count = np.sum(np.multiply(img_, mask_) > thre_white)
    m_points_to_analyse = np.where((mask_ > 0) == True)
    m_left_ = np.amin(m_points_to_analyse[1])
    m_right_ = np.amax(m_points_to_analyse[1])
    weight = (m_right_ - m_left_)
    ack = False
    if count > weight:
        ack = True
    return ack


## It removes the candidates that are not in or above the rpe layer.
    #  @param masks_ candidates.
    #  @param img_ original image.
    #  @param depth_to_use number of lines to be analysed.
    #  @param threshold_white threshold to determine what is a white pixel.
    #  @return a new mask with the remaining candidates.



def remove_above_rpe(masks_, img_, depth_to_use=5, threshold_white=200):
    new_masks = []
    for actual_mask in masks_:
        m_points_to_analyse = np.where((actual_mask > 0) == True)
        m_bottom_ = m_points_to_analyse[0][len(m_points_to_analyse[1]) - 1]
        if m_bottom_ + depth_to_use < img_.shape[0]:
            actual_mask_aux = np.roll(actual_mask, depth_to_use, axis=0)
            actual_mask_aux = actual_mask - actual_mask_aux
            actual_mask_aux[np.where(actual_mask_aux > 0)] = 0
            actual_mask_aux[np.where(actual_mask_aux < 0)] = 1
            if detect_possible_rpe(img_, actual_mask_aux, threshold_white) == True:
                new_masks.append(actual_mask)
    return new_masks



## It generates all masks/candidates and applies the selection of the candidates.
    #  @param grabcut_image image with no background noise.
    #  @param cropped_image image with no background noise and without the region that contains only background
    #  @param grabcut_mask mask generated to select foreground pixels.
    #  @param maximum_total_pixels percentage of the total pixel that a region is allowed to have.
    #  @param threshold_black_pixel threshold value for black pixel.
    #  @param minimum_black_density minimum black pixel density required for a certain region.
    #  @param minimum_pixel_density minimum pixel density required for a certain region.
    #  @return a mask with remaining candidates.

def find_SNF_mask(grabcut_image, cropped_image, grabcut_mask, maximum_total_pixels=0.035, threshold_black_pixel=30, minimum_black_density=0.5, minimum_pixel_density=0.3):

    gray_img = utils.rgb_2_gray(grabcut_image)
    # applying grabcut mask to remove the background region
    gray_img = np.multiply(gray_img, grabcut_mask)
    hist = get_hist(gray_img)
    # print("mask before remove large region " + str(hist[1].__len__() ))
    # hist = remove_large_regions(hist, oct_pixels, maximum_total_pixels)
    masks = creating_masks_from_histogram(gray_img, hist)
    print("mask before remove low density black pixel " + str(masks.__len__() ))
    masks = remove_low_density_black_pixel(masks, cropped_image, minimum_black_density, threshold_black_pixel)
    ##masks = remove_low_density_candidates(masks, minimum_pixel_density)
    print("mask before remove in border " + str(masks.__len__() ))
    masks = remove_in_border(masks, cropped_image)
    print("mask before remove above ilm " + str(masks.__len__() ))
    masks = remove_above_ilm(masks, cropped_image)
    print("mask before remove above rpe " + str(masks.__len__() ))
    masks = remove_above_rpe(masks, cropped_image)
    print("mask at the end " + str(masks.__len__() ))
    return masks


## It detects SNF in the cropped image.
    #  @param grabcut_image image with no background noise.
    #  @param cropped_image image with no background noise and without the region that contains only background
    #  @param grabcut_mask mask generated to select foreground pixels.
    #  @param maximum_total_pixels percentage of the total pixel that a region is allowed to have.
    #  @param threshold_black_pixel threshold value for black pixel.
    #  @param minimum_black_density minimum black pixel density required for a certain region.
    #  @param minimum_pixel_density minimum pixel density required for a certain region.
    #  @return True if cropped image contains SNF. False otherwise.


def detect_SNF(grabcut_image, cropped_image, grabcut_mask, maximum_total_pixels=0.035, threshold_black_pixel=30, minimum_black_density=0.5, minimum_pixel_density=0.3):
    masks = find_SNF_mask(grabcut_image, cropped_image, grabcut_mask, maximum_total_pixels=0.035, threshold_black_pixel=30, minimum_black_density=0.5, minimum_pixel_density=0.3)
    if masks.__len__() > 0:
        ack = True
    else:
        ack = False
    return ack
