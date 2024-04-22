import cv2
import numpy as np

fname = input("Entering image file name:")
img  = cv2.imread(fname)

b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]

rgb_img = np.hstack((r,g,b))
cv2.imshow("RGB-images",rgb_img)
cv2.waitKey(0)
cv2.destroyAllWindows()