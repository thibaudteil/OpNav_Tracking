###############################################################
#
#   This test script implements a few of the methods in the
#   online openCV tutorial http://docs.opencv.org/
#
#


import glob
from cv2 import *
import cv2
import imageio
from Translations import NumFrames

# Make path to save images
file_path = '../../MovImgs'
MovieName = 'MovingCraters'
extension = '.jpg'

images = []
Frames = range(1,NumFrames+1)
for filename in Frames:
    images.append(imageio.imread(file_path + '/' + str(filename) + extension))
imageio.mimsave( MovieName + '.gif', images)