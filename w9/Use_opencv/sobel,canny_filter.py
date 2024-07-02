#sobel의 어떤 단점을 보완한것이 canny인지 알아보자 
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)
        canny = cv2.Canny(gray, 50, 150)
        cframe = np.hstack((gray, sobel, canny))
        cv2.imshow('High-pass filtered results', cframe)

    key = cv2.waitKey(33)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()