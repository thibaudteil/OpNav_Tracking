import os
import cv2
import numpy as np

from simple_detectors import *

root = ".."
img_dir = "images"
img_name = "moon_1.jpg"
img_path = os.path.join(root, img_dir, img_name)

img = cv2.imread(img_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
mean_pixels = img_gray.mean()
img_new = (img_gray - mean_pixels).clip(min=0)
img_new = img_new.astype(np.uint8)
img_new[img_new > 128] = 255
img_new[img_new < 128] = 0

circles = cv2.HoughCircles(img_new, cv2.HOUGH_GRADIENT, 1, 75, param1=50, param2=30, minRadius=0, maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img_new, (i[0], i[1]), i[2], (255), 2)
    # draw the center of the circle
    cv2.circle(img_new, (i[0], i[1]), 2, (255), 3)

cv2.imwrite('test_1.jpg', img_new)
if True:
    cv2.imshow('Detected circles',img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()
