# gain 도 변경해보자
# kernel도 극으로 변경
# 4 개의 세트 noise가 많은것과 필터 2개씩 매칭 

import cv2
import numpy as np

ksize = int(input('Enter kernel size:'))
kernel = np.ones((ksize, ksize), np.float32)/(ksize*ksize)
ngain = float(input('Enter noise gain:'))
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
        noisy = np.clip(Y + np.random.random((height, width))*ngain, 0, 255).astype(np.uint8)
        filtered = cv2.filter2D(noisy, -1, kernel)
        cnoisy = cv2.cvtColor(cv2.merge((noisy, Cr, Cb)), cv2.COLOR_YCrCb2BGR)
        cfiltered = cv2.cvtColor(cv2.merge((filtered, Cr, Cb)), cv2.COLOR_YCrCb2BGR)
        cframe = np.hstack((frame, cnoisy, cfiltered))
        cv2.imshow('Original, Noisy, Filtered', cframe)

    key = cv2.waitKey(33)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
