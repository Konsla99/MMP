
#pg 17
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
        max_y = np.max(gray)
        min_y = np.min(gray)
        cgray = (255.*(gray-min_y)/(max_y-min_y)).astype(np.uint8)
        cimg =np.hstack((gray, cgray))
        cv2.imshow('Histogram stretching', cimg)
        key = cv2.waitKey(33)
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()