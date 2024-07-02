import cv2
import random
import numpy as np

ksize = int(input('Enter kernel size:'))  # 가우시안 커널 크기 입력 (홀수를 권장, 예: 3, 5, 7)
rat_noise = float(input('Enter frequency of noise:'))  # 노이즈 비율 입력
width = 560
height = 720

# 이미지 파일 경로 직접 설정
image_path = 'D:/code/mmp/w9/data/test.jpg'  # 이미지 경로를 여기에 입력하세요

# 이미지 파일 읽기 및 크기 조정
frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
if frame is None:
    print("Error: Image could not be read. Check the path.")
    exit()
frame = cv2.resize(frame, (width, height))  # 이미지를 지정된 width와 height로 조정

t_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = cv2.split(t_frame)
num_noise = int(width*height*rat_noise)

# Salt noise 추가
for i in range(num_noise):
    y = random.randint(0, height-1)
    x = random.randint(0, width-1)
    Y[y][x] = 255  # 흰 점 (Salt noise)

# 가우시안 필터 적용 (커널 크기와 표준 편차를 입력으로 받음, 표준 편차가 0이면 자동 계산됨)
filtered = cv2.GaussianBlur(Y, (ksize, ksize), 0)
cnoisy = cv2.cvtColor(cv2.merge((Y, Cr, Cb)), cv2.COLOR_YCrCb2BGR)
cfiltered = cv2.cvtColor(cv2.merge((filtered, Cr, Cb)), cv2.COLOR_YCrCb2BGR)
cframe = np.hstack((frame, cnoisy, cfiltered))
cv2.imshow('Original, Noisy, Gaussian-filtered', cframe)

# 'q' 키를 누르면 종료
key = cv2.waitKey(0)  # 대기 시간을 무한대로 설정하여 사용자가 키를 누를 때까지 기다립니다.
if key == ord('q'):
    cv2.destroyAllWindows()
