## @package MyProject
#  Main py file that uses MyTester as interface to detect if an OCT image has or does not have SNF.
# @author Thiago Pincinato and Tamara Melle

from scipy.misc import imread
import matplotlib.pyplot as plt
import glob
import MyTester as tester


# opening all images
img_to_analyse = []
border_X_left, border_X_right, border_Y_top, border_Y_bottom = 48, 15, 15, 45

for img in glob.glob("TestData/handout/*.png"):
    n = imread(img)
    img_to_analyse.append(n[border_Y_top:n.shape[0] - border_Y_bottom, border_X_left:n.shape[1] - border_X_right])


sigma = 1
k = 200
min_ = 250
tester.test_detection_SNF(img_to_analyse, sigma, k, min_)
