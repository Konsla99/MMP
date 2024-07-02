#salt noise에 적용해볼것

import cv2
import numpy as np
width = 320
height = 240
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
while True:
    ret, frame = cap.read()
    if ret:
        t_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(t_frame)
        blur = cv2.GaussianBlur(Y, (0, 0), 2)
        filtered_Y = np.clip(2.0*Y - blur, 0, 255).astype(np.uint8)
        cfiltered = cv2.cvtColor(cv2.merge((filtered_Y, Cr, Cb)), cv2.COLOR_YCrCb2BGR)
        cframe = np.hstack((cv2.flip(frame,1), cv2.flip(cfiltered,1)))
        cv2.imshow('Original, Unsharp-mask', cframe)

    key = cv2.waitKey(33)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()