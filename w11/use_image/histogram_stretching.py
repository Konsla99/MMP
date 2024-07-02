import cv2
import numpy as np

# 사용자로부터 이미지 경로 입력받기
image_path = 'D:/code/mmp/W10/w10_img/low.jpg'

# 이미지를 읽고 크기 조정
img = cv2.imread(image_path)
img = cv2.resize(img, (320, 240))

while True:
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        max_y = np.max(gray)
        min_y = np.min(gray)
        # 히스토그램 스트레칭 적용
        cgray = (255. * (gray - min_y) / (max_y - min_y)).astype(np.uint8)
        cimg = np.hstack((gray, cgray))
        cv2.imshow('Histogram stretching', cimg)
        key = cv2.waitKey(33)
        if key == ord('q'):
            break

cv2.destroyAllWindows()
