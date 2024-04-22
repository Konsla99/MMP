import cv2
import numpy as np

cap = cv2.VideoCapture(0,cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)

while True:
    ret,frame = cap.read()
    if ret == True:  
        b = frame[:,:,0]
        g = frame[:,:,1]  
        r = frame[:,:,2]
        rgb_frmae = np.hstack((r,g,b))
        cv2.imshow("RGB-images",rgb_frmae)

    key = cv2.waitKey(33)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

