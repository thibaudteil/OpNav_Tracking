'''
    Wrappers for a few of the default detectors OpenCV offers
'''

import cv2
import os
import numpy as np

from matplotlib import pyplot as plt

# FAST Algorithm for Corner Detection
# Returns an image with dectected features marked
def fast(img_path, threshold=10, detect_type=cv2.FAST_FEATURE_DETECTOR_TYPE_5_8, nonmaxSuppression=1):

    # Load image
    img = cv2.imread(img_path, 0)

    # Initiate FAST object
    fast = cv2.FastFeatureDetector_create(threshold=threshold, type=detect_type)

    # Find and draw Key Points
    kp = fast.detect(img, None)
    output = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

    return output


# Harris Corner Detection
# Returns an image with dectected features marked
def harris(img_path):

    # Load image
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray,2,3,0.04)

    # Result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0,0,255]

    return img
