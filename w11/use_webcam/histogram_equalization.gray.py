
#pg 21

import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
while True:
    ret, img = cap.read()
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equ_gray = cv2.equalizeHist(gray)
        cimg =np.hstack((gray, equ_gray))
        cv2.imshow('Histogram stretching', cimg)
        key = cv2.waitKey(33)
        if key == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()