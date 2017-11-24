###############################################################
#
#   This test script implements a few of the methods in the
#   online openCV tutorial http://docs.opencv.org/
#
#


import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import time
from fractions import gcd
def lcm(a,b): return abs(a * b) / gcd(a,b) if a and b else 0

# Make path to save images
file_path = '../../MovImgs/'

directory = os.path.dirname(file_path)
if not os.path.exists(directory):
    os.makedirs(directory)


# Put show_plots to true in order to have plt show plots
show_plots = False
imPath = '../images/'
imName = 'Moon_test'
imExt = '.jpg'


moon = cv2.imread(imPath + imName + imExt)
rows,cols, thing = moon.shape

NumFrames = 400
for i in range(NumFrames//2):
    frameName = str(i+1)
    M = np.float32([[1,0,cols - 4*i*int(cols/NumFrames)],[0,1,rows - 4*i*int(rows/NumFrames)]])
    dst = cv2.warpAffine(moon,M,(cols,rows))

    # cv2.imshow('frame' + str(i), dst)
    cv2.imwrite(file_path + '/' + frameName + '.jpg', dst)

    cv2.waitKey(0)
    # cv2.destroyAllWindows()

for i in range(NumFrames//2, NumFrames):
    frameName = str(i+1)
    M = np.float32([[1,0,cols - 4*NumFrames*int(cols/NumFrames) + 4*i*int(cols/NumFrames)],[0,1,rows - 4*NumFrames*int(rows/NumFrames) + 4*i*int(rows/NumFrames)]])
    dst = cv2.warpAffine(moon,M,(cols,rows))

    # cv2.imshow('frame' + str(i), dst)
    cv2.imwrite(file_path + '/' + frameName + '.jpg', dst)

    cv2.waitKey(0)
    # cv2.destroyAllWindows()
