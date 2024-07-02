import cv2
import numpy as np

width = 320
height = 240
k_lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
k_lap_e = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

# 이미지 파일 경로 직접 설정
image_path = 'D:/code/mmp/w9/data/test.jpg'  # 이미지 경로

# 이미지 파일 읽기 및 크기 조정
frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
if frame is None:
    print("Error: Image could not be read. Check the path.")
    exit()
frame = cv2.resize(frame, (width, height))  # 이미지를 지정된 width와 height로 조정

t_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = cv2.split(t_frame)
Yf_lap = cv2.filter2D(Y, -1, k_lap)
Yf_lap_e = cv2.filter2D(Y, -1, k_lap_e)
img_lap = cv2.cvtColor(cv2.merge((Yf_lap, Cr, Cb)), cv2.COLOR_YCrCb2BGR)
img_lap_e = cv2.cvtColor(cv2.merge((Yf_lap_e, Cr, Cb)), cv2.COLOR_YCrCb2BGR)
cframe = np.hstack((frame, img_lap, img_lap_e))
cv2.imshow('High-pass filtered results', cframe)

# 'q' 키를 누르면 종료
key = cv2.waitKey(0)  # 대기 시간을 무한대로 설정
if key == ord('q'):
    cv2.destroyAllWindows()
