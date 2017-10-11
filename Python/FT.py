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
img_in = cv2.imread(imPath + imName + imExt,0)

###################################################
#Make image square

height, width = img_in.shape
if height>width:
    startH = height - width - (height-width)/2
    endH = height - (height-width)/2
    img = img_in[0:startH:endH-1, 0:width-1]
if height<width:
    startW = -height + width + (height-width)/2
    endW = width + (height-width)/2
    cimg = img_in[0:height, startW-1:endW-1]

# cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

#Blurring by averaging over 25 pixels (5,5) arrays
f = np.fft.fft2(cimg)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))


# Plot blurred images
plt.subplot(121),plt.imshow(cimg),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum),plt.title('FFT')
plt.xticks([]), plt.yticks([])
plt.show()
plt.close()

rows, cols = cimg.shape
crow,ccol = rows/2 , cols/2
fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.subplot(121),plt.imshow(cimg, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back)
plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

plt.show()

# The forth argument is the min distance between centers of detected circles
# Have to make it large to not find too many
img_circ = img_back.astype('uint8')
circles = cv2.HoughCircles(img_circ,cv2.HOUGH_GRADIENT,1,30,
                            param1=50,param2=30,minRadius=0,maxRadius=150)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    cv2.circle(img_circ,(i[0],i[1]),i[2],(0,255,0),2)
    cv2.circle(img_circ,(i[0],i[1]),2,(0,0,255),3)

cv2.imwrite(imPath+imName+'_circ'+ imExt, img_back)
if True:
    cv2.imshow('Detected circles',img_back)
cv2.waitKey(0)
cv2.destroyAllWindows()
