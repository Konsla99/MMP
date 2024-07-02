import cv2
import numpy as np

image_path = 'D:/code/mmp/w12/Hough/baseball.jpg'
img = cv2.imread(image_path)

# 이미지를 원하는 크기로 리사이즈하며, 비율을 유지합니다.
height, width = img.shape[:2]
new_height = 400
new_width = int((new_height / height) * width)
img = cv2.resize(img, (new_width, new_height))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)

def on_trackbar(pos):
    dp = cv2.getTrackbarPos('dp', 'Trackbars') / 10.0
    minDist = cv2.getTrackbarPos('minDist', 'Trackbars')
    param1 = cv2.getTrackbarPos('param1', 'Trackbars')
    param2 = cv2.getTrackbarPos('param2', 'Trackbars')
    minRadius = cv2.getTrackbarPos('minRadius', 'Trackbars')
    maxRadius = cv2.getTrackbarPos('maxRadius', 'Trackbars')

    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    dst = img.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # 원 둘레에 초록색 원 그리기
            cv2.circle(dst, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # 원 중심점에 빨강색 원 그리기
            cv2.circle(dst, (i[0], i[1]), 2, (0, 0, 255), 5)

    merged = np.hstack((img, dst))
    cv2.imshow('Trackbars', merged)

# 윈도우 생성 및 초기 이미지 표시
cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Trackbars', new_width * 2, new_height + 50)  # 윈도우 크기 조정

# 트랙바 생성
cv2.createTrackbar('dp', 'Trackbars', 10, 50, on_trackbar)  # dp 값을 1.0 ~ 5.0 단위로 조정할 수 있게 설정
cv2.setTrackbarPos('dp', 'Trackbars', 12)  # 초기값 설정 (1.2)

cv2.createTrackbar('minDist', 'Trackbars', 1, 200, on_trackbar)
cv2.setTrackbarPos('minDist', 'Trackbars', 60)

cv2.createTrackbar('param1', 'Trackbars', 0, 500, on_trackbar)
cv2.setTrackbarPos('param1', 'Trackbars', 100)

cv2.createTrackbar('param2', 'Trackbars', 1, 300, on_trackbar)
cv2.setTrackbarPos('param2', 'Trackbars', 30)

cv2.createTrackbar('minRadius', 'Trackbars', 0, 200, on_trackbar)
cv2.setTrackbarPos('minRadius', 'Trackbars', 10)

cv2.createTrackbar('maxRadius', 'Trackbars', 0, 200, on_trackbar)
cv2.setTrackbarPos('maxRadius', 'Trackbars', 100)

# 초기 콜백 호출
on_trackbar(0)

cv2.waitKey(0)
cv2.destroyAllWindows()
