import cv2
import numpy as np

img_path = 'D:/code/mmp/w12/Hough/baseball_field3.jpg'
img = cv2.imread(img_path)
Thres = int(input("Enter accumulation threshold:"))

h=480
w=320

rimg = cv2.resize(img,(h,w))
img2 = rimg.copy()



Y, Cr, Cb = cv2.split(cv2.cvtColor(rimg, cv2.COLOR_BGR2YCrCb))


# # 8방향 라플라시안 필터 커널
# k_lap_e = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# # 8방향 라플라시안 필터 적용
# H_edges = cv2.filter2D(Y, -1, k_lap_e)
# _, H_edges = cv2.threshold(H_edges, 30, 255, cv2.THRESH_BINARY)

edges = cv2.Canny(Y, 50, 150)
edge_color = cv2.cvtColor(cv2.merge((edges, Cr, Cb)), cv2.COLOR_YCrCb2BGR)

# 변환 및 추출된 edge 의 화면표시
# Rho =  선의 해상도에 영향을 미침
# theta 각 도의 해상도 작을수록 정밀한 값
# thres =  선이 지나는 최소한의 점의 개수
lines = cv2.HoughLines(edges, 1, np.pi/180, Thres)
for line in lines:
    r,theta = line[0]
    tx, ty = np.cos(theta), np.sin(theta)
    x0, y0 = tx*r, ty*r

    x1, y1 = int(x0 + w*(-ty)), int(y0 + h * tx)
    x2, y2 = int(x0 - w*(-ty)), int(y0 - h * tx)

    cv2.line(img2, (x1, y1), (x2, y2), (255, 0, 0), 2)
merged = np.hstack((rimg, edge_color, img2))
cv2.imshow('hough lines', merged)
cv2.waitKey()
cv2.destroyAllWindows()