import cv2
import numpy as np

img_path = 'D:/code/mmp/w12/Hough/baseball_field3.jpg'
img = cv2.imread(img_path)
Thres = int(input("Enter accumulation threshold:"))

# 원본 이미지 사용
img2 = img.copy()

# YCrCb로 변환 및 채널 분리
Y, Cr, Cb = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))

# Canny edge detection
edges = cv2.Canny(Y, 100,250)
edge_color = cv2.cvtColor(cv2.merge((edges, Cr, Cb)), cv2.COLOR_YCrCb2BGR)

# Hough Line Transform
lines = cv2.HoughLines(edges, 1, np.pi / 90, Thres)
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
        x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)

        # 검출된 선이 실제로 edge에 있는지 확인
        if edges[min(max(y1, 0), img.shape[0]-1), min(max(x1, 0), img.shape[1]-1)] > 0 or edges[min(max(y2, 0), img.shape[0]-1), min(max(x2, 0), img.shape[1]-1)] > 0:
            cv2.line(img2, (x1, y1), (x2, y2), (255, 0, 0), 2)

# 결과 이미지 병합 및 출력
merged = np.hstack((img, edge_color, img2))
cv2.imshow('hough lines', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
