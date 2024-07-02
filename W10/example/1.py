import cv2
import numpy as np

width = 320
height = 240
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
# ret는 cv2.videocapture이 ret과 img를 반환
# img = 이미지 그 자체
# ret =  프레임 읽기 작업을 성공했는지 나타내는 플레그
while True:
    ret , img = cap.read()
    if ret:
        gray = cv2.cvtColor(img , cv2.COLOR_BAYER_BG2GRAY)
        F = np.fft.fft2(gray)
        Fshift = np.fft.fftshift(F)
        mag_spc = np.clip (( 20*np.log (np.abs(Fshift ))), 0,255).astype(np.uint8)
        cframe = np.hstack ((gray , mag_spc))
        cv2.imshow('2D-FFT', cframe)
    
        key = cv2.waitKey(33)
        if key == ord ('q'):
            break
cap.release
cv2.destroyAllWindows()