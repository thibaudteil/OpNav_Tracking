###############################################################
#
#   This test script implements a few of the methods in the
#   online openCV tutorial http://docs.opencv.org/
#
#


import numpy as np
from matplotlib import pyplot as plt
import cv2
import time
from fractions import gcd
def lcm(a,b): return abs(a * b) / gcd(a,b) if a and b else 0

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
# bluring image in different ways
###############################################################
time1 = time.time()

img = cv2.imread(imPath + imName + imExt,0)
width, height = img.shape[:2]
blur_coef = gcd(width, height)**2

## Gaussian Blur doesn't work so well
# blur_img = cv2.GaussianBlur(img,(blur_coef,blur_coef),0)
## Bilateral Blur doesn't work so well either and is slow
# blur_img = cv2.bilateralFilter(img, blur_coef, 0, blur_coef )

## Median Blur is best atm
blur_img = cv2.medianBlur(img,blur_coef)

###############################################################
# do threshold for image and it's inverse, then add them
# Can use:
# THRESH_BINARY
# THRESH_TOZERO
ret,cimg1 = cv2.threshold(blur_img,127,255,cv2.THRESH_TOZERO)
ret,cimg2 = cv2.threshold(blur_img,127,255,cv2.THRESH_TOZERO_INV)


# Add two images with weighting
alpha = 0.5
beta = ( 1.0 - alpha )
cimg = cv2.addWeighted( cimg1, alpha, cimg2, beta, 0.0)
cv2.imshow('Adding', cimg)

## Adaptive thresholding was also an option
# cimg = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

# The forth argument is the min distance between centers of detected circles
# Have to make it large to not find too many
circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,30,
                            param1=50,param2=30,minRadius=0,maxRadius=150)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

time2 = time.time()
print time2 - time1
cv2.imwrite(imPath+imName+'_circ'+ imExt, cimg)
if True:
    cv2.imshow('Detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()