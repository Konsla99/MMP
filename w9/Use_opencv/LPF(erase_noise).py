#평균 필터의 노이즈 제거원리는 무엇인가?
# 조사해볼것!!
# gain 도 변경해보자
# kernel도 극으로 변경

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
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        noisy = np.clip(img + np.random.random((height, width))*ngain, 0, 255).astype(np.uint8)
        filtered_img = cv2.filter2D(noisy, -1, kernel)
        cframe = np.hstack((img, noisy, filtered_img))
        cv2.imshow('Original, Noisy, Filtered', cframe)

    key = cv2.waitKey(33)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()