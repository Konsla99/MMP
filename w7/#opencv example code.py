#opencv example code
# must be global interpreter

import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, frame = cap.read()
    b,g,r = np.average(frame,axis = (0,1))
    text = ("Avg. B:%3d,G%3d,R:%3d"%(b,g,r))
    cv2.putText(frame,text,(5,20),font,0.6,(255,180,180),2)
    if ret == True:
        cv2.imshow("Camera",frame)

    key = cv2.waitKey(33)
    if key == ord('q'):
        break    

cap.release()
cv2.destroyAllWindows()