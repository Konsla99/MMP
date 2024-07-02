import cv2
import numpy as np

# 사용자로부터 이미지 경로 입력받기
image_path = 'D:/code/mmp/W10/w10_img/low.jpg'

# 이미지를 읽고 크기 조정
# 이미지를 읽고 크기 조정
img = cv2.imread(image_path)
img = cv2.resize(img, (320, 240))

while True:
    if img is not None:
        # 이미지를 회색조로 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 히스토그램 평활화 적용
        equ_gray = cv2.equalizeHist(gray)
        cimg = np.hstack((gray, equ_gray))
        cv2.imshow('Histogram equalization', cimg)
        key = cv2.waitKey(33)
        if key == ord('q'):
            break

cv2.destroyAllWindows()
