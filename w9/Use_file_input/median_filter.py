# salt noise를 가우시안 필터와 비교해 볼것
# 검은 점을 찍어서 인위적인 점도 지워지는지 확인
# 255 = 흰점 salt  , 0 은 검은점 pepper
# 밝기에 따른 문제를 적자!
# ration 입력을 다시한번 확인하자 정규화되어 있으므로 0.1정도

import cv2
import random
import numpy as np

ksize = int(input('Enter kernel size:'))
rat_noise = float(input('Enter frequency of noise:'))
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
    Y[y][x] = 0  # 흰 점 (Salt noise)

filtered = cv2.medianBlur(Y, ksize)
cnoisy = cv2.cvtColor(cv2.merge((Y, Cr, Cb)), cv2.COLOR_YCrCb2BGR)
cfiltered = cv2.cvtColor(cv2.merge((filtered, Cr, Cb)), cv2.COLOR_YCrCb2BGR)
cframe = np.hstack((frame, cnoisy, cfiltered))
cv2.imshow('Original, Noisy, Median-filtered', cframe)

# 'q' 키를 누르면 종료
key = cv2.waitKey(0)  # 대기 시간을 무한대로 설정하여 사용자가 키를 누를 때까지 기다립니다.
if key == ord('q'):
    cv2.destroyAllWindows()
