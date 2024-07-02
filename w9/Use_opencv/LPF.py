# 해상도 늘려보기 cap.set부분
# cv2. filter2d의  bordertype 변경해보기
# default = border_reflect_101 
# 이건 LPF에서 가장 성능이 좋다
# Border_constant 는 HPF에서 좋다 그러므로 이거 해보자
# BOrder_Replicate는 임의의 값을 패딩 한다 총 3개의 시도

import cv2
import numpy as np

ksize = int(input('Enter kernel size:'))
kernel = np.ones((ksize, ksize), np.float32)/(ksize*ksize)

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

while True:
    ret, frame = cap.read()
    if ret:
        filtered_frame = cv2.filter2D(frame, -1, kernel)
        cframe = np.hstack((frame, filtered_frame))
        cv2.imshow('Image', cframe)

    key = cv2.waitKey(33)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()