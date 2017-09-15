###############################################################
#
#   This test script implements a few of the methods in the
#   online openCV tutorial http://docs.opencv.org/
#
#


import numpy as np
from matplotlib import pyplot as plt
import cv2

# Put show_plots to true in order to have plt show plots
show_plots = False
imPath = '../images/'
imName = 'Moon_test'
imExt = '.jpg'

# Load the moon images
moon = cv2.imread(imPath + imName + imExt)

#Blurring by averaging over 25 pixels (5,5) arrays
kernel = np.ones((5,5),np.float32)/25
moon_blurAvg = cv2.filter2D(moon,-1,kernel)

#Gaussian blurring
moon_blurGauss = cv2.GaussianBlur(moon,(5,5),0)

# Plot blurred images
plt.subplot(121),plt.imshow(moon),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(moon_blurAvg),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
if show_plots:
    plt.show()
plt.close()

plt.subplot(121),plt.imshow(moon),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(moon_blurGauss),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
if show_plots:
    plt.show()
plt.close()



###############################################################
# Find circles on an image with Hough Circle Transform
#
# This method needs testing when it comes to performance, it seems
# that an image larger than a dozen KB takes a long time
# it actuall finds many more circles than should be found it seems
# Need to understand how to manage sensitivites.
# The plots look like garbage (circle finding is crap)

img = cv2.imread(imPath + imName + imExt,0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

# The forth argument is the min distance between centers of detected circles
# Have to make it large to not find too many
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,75,
                            param1=50,param2=30,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imwrite(imPath+imName+'_circ'+ imExt, cimg)
if True:
    cv2.imshow('Detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()