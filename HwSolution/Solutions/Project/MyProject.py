from scipy.misc import imread
import matplotlib.pyplot as plt
import glob
import MyTester as tester



# opening all images
img_NoSRF = []
img_SRF = []
border_X_left, border_X_right, border_Y_top, border_Y_bottom = 48, 15, 15, 45

for img in glob.glob("TrainData/NoSRF/*.png"):
    n = imread(img)
    img_NoSRF.append(n[border_Y_top:n.shape[0] - border_Y_bottom, border_X_left:n.shape[1] - border_X_right])

for img2 in glob.glob("TrainData/SRF/*.png"):
    n = imread(img2)
    img_SRF.append(n[border_Y_top:n.shape[0] - border_Y_bottom, border_X_left:n.shape[1] - border_X_right])


#tester.test_grab_cut(img_SRF[3:4])
#plt.show()

sigma = 1 #3 # 0.8 #1.0 #3
k = 200# 150# 250 #150.0
min_ = 250# 50 200 #200 #50  #

#tester.test_seg_k_m_sigma(img_SRF[7:9])
#tester.test_segmentation(img_SRF, sigma, k, min_)
#tester.test_segmentation(img_SRF[7:9], sigma, k, min_)
# showing graphs

#tester.test_detection(img_NoSRF[10:11], sigma, k, min_)
# Corrected not detected
# 0, 1, 2, 3,6,7,8,9,11,12, 13, 14,
# Actual errors
# 4,5,10

tester.test_dection_SNF(img_SRF[14:], sigma, k, min_)

# Correct detected
# 2,3,5,6,7,8,9,14
# 11! one region detected more than it should

# Actual error (! can be corrected, other demand change in segmentation
# 0, 1, 4, 10, 12, 13
#
#TP = tester.test_detection_positive_rate(img_SRF, sigma, k, min_)
#TN = tester.test_detection_negative_rate(img_NoSRF, sigma, k, min_)
#print("Detection of SNF:  " + str(TP))
#print("Detection of NoSNF:  " + str(TN))
plt.show()
