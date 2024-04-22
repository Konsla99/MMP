import cv2
import numpy as np

fname = input("Entering image file name:")
img  = cv2.imread(fname)
sy,sx,_ = img.shape

wimg = np.zeros((sy,sx,3), dtype=np.uint8)
wimg[:,:,0] = img[:,:,0]
cv2.imwrite('blue.jpg',wimg)

wimg.fill(0)
wimg[:,:,1] = img[:,:,1]
cv2.imwrite('green.jpg',wimg)


wimg.fill(0)
wimg[:,:,2] = img[:,:,2]
cv2.imwrite('red.jpg',wimg)
