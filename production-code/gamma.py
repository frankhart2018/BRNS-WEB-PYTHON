import numpy as np
import argparse
import cv2
import os

def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / float(gamma)
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	return cv2.LUT(image, table)

def gamma_correction(gamma, filename):
    original = cv2.imread(filename)
    adjusted = adjust_gamma(original, gamma=gamma)
    cv2.putText(adjusted, "gamma={}".format(gamma), (10, 30),
    	cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    file_loc = "static/gamma_corrected/" + os.path.basename(filename)
    cv2.imwrite(file_loc, adjusted)

    return file_loc
