import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

img = cv2.imread('static/colorized/sttest1.jpg')

img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imwrite('static/test.jpg', img1)

#
# r = np.divide(R, all_channels)
# g = G/(R+G+B)
# b = B/(R+G+B)
#
# H = np.zeros(r.shape)
