import cv2
import numpy as np

fname = input("Entering image file name:")
src  = cv2.imread(fname)
dst = cv2.cvtColor(src,cv2.COLOR_BGR2YCrCb)
cv2.imwrite('Y.jpg',dst[:,:,0])
cv2.imwrite('Cr.jpg',dst[:,:,1])
cv2.imwrite('Cb.jpg',dst[:,:,2])