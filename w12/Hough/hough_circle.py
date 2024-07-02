import cv2
import numpy as np

image_path = 'D:/code/mmp/w12/Hough/baseball.jpg'
img = cv2.imread(image_path)
Thres = int(input("Enter canny threshold:"))
img2 = img.copy()

Y, Cr, Cb = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
blur = cv2.GaussianBlur(Y, (3,3), 0)
# dp= n일경우 n배 작은 이미지에서 원을 검출
#minDist는 원 중심들 사이 최소거리 작으면 겹치는 원들이 많이 검출

circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2, 80, None, Thres)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(img2,(i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(img2, (i[0], i[1]), 2, (0,0,255), 5)

merged = np.hstack((img, img2))
cv2.imshow('HoughCircles', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()

## page 14