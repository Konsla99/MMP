# salt noise를 가우시안 필터와 비교해 볼것
# 검은 점을 찍어서 인위적인 점도 지워지는지 확인
# 255 = 흰점 salt  , 0 은 검은점 pepper

import cv2
import random
import numpy as np
ksize = int(input('Enter kernel size:'))
rat_noise = float(input('Enter frequency of noise:'))
width = 320
height = 240
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
num_noise = int(width*height*rat_noise)
while True:
    ret, frame = cap.read()
    if ret:
        t_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(t_frame)
        for i in range(num_noise):
            y=random.randint(0, height-1)
            x=random.randint(0, width-1)
            Y[y][x] = 0
        filtered = cv2.medianBlur(Y, ksize)
        cnoisy = cv2.cvtColor(cv2.merge((Y, Cr, Cb)), cv2.COLOR_YCrCb2BGR)
        cfiltered = cv2.cvtColor(cv2.merge((filtered, Cr, Cb)), cv2.COLOR_YCrCb2BGR)
        cframe = np.hstack((frame, cnoisy, cfiltered))
        cv2.imshow('Original, Noisy, Median-filtered', cframe)

    key = cv2.waitKey(33)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()