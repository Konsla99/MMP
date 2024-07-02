import cv2
import numpy as np

ksize = int(input('Enter kernel size:'))  # 가우시안 필터 크기 입력, 일반적으로 홀수 사용
ngain = float(input('Enter noise gain:'))  # 노이즈 게인 입력
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
        noisy = np.clip(Y + np.random.random((height, width)) * ngain, 0, 255).astype(np.uint8)
        # 가우시안 필터 적용, sigmaX를 0으로 설정하여 자동 계산하도록 함
        filtered = cv2.GaussianBlur(noisy, (ksize, ksize), 0)
        cnoisy = cv2.cvtColor(cv2.merge((noisy, Cr, Cb)), cv2.COLOR_YCrCb2BGR)
        cfiltered = cv2.cvtColor(cv2.merge((filtered, Cr, Cb)), cv2.COLOR_YCrCb2BGR)
        cframe = np.hstack((frame, cnoisy, cfiltered))
        cv2.imshow('Original, Noisy, Gaussian-filtered', cframe)

    key = cv2.waitKey(33)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
