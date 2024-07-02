import cv2
import numpy as np


img_path = 'D:/code/mmp/w12/Hough/baseball_field3.jpg'
img = cv2.imread(img_path)
Thres = int(input("Enter accumulation threshold:"))

h=320
w=240
rimg = cv2.resize(img,(h,w))
img2 = rimg.copy()

Y, Cr, Cb = cv2.split(cv2.cvtColor(rimg, cv2.COLOR_BGR2YCrCb))

# s_edges = cv2.Sobel(Y, cv2.CV_8U, 1, 0, 3)
# k_lap_e = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# 8방향 라플라시안 필터 적용
# H_edges = cv2.filter2D(Y, -1, k_lap_e)
# _, H_edges = cv2.threshold(H_edges, 30, 255, cv2.THRESH_BINARY)


edges = cv2.Canny(Y, 150, 250)
edge_color = cv2.cvtColor(cv2.merge((edges, Cr, Cb)), cv2.COLOR_YCrCb2BGR)


lines = cv2.HoughLinesP(edges, 1, np.pi/180, Thres, None, 5,10)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img2, (x1,y1), (x2, y2), (0, 255, 0), 2)
merged = np.hstack((rimg, edge_color, img2))
cv2.imshow('Probability hough line', merged)
cv2.waitKey()
cv2.destroyAllWindows()
##page 11qq