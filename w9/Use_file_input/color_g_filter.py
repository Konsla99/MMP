import cv2
import numpy as np

ksize = int(input('Enter kernel size:'))  # 가우시안 필터 크기 입력, 일반적으로 홀수 사용
ngain = float(input('Enter noise gain:'))  # 노이즈 게인 입력
width = 320
height = 240

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
noisy = np.clip(Y + np.random.random((height, width)) * ngain, 0, 255).astype(np.uint8)
# 가우시안 필터 적용, sigmaX를 0으로 설정하여 자동 계산하도록 함
filtered = cv2.GaussianBlur(noisy, (ksize, ksize), 0)
cnoisy = cv2.cvtColor(cv2.merge((noisy, Cr, Cb)), cv2.COLOR_YCrCb2BGR)
cfiltered = cv2.cvtColor(cv2.merge((filtered, Cr, Cb)), cv2.COLOR_YCrCb2BGR)
cframe = np.hstack((frame, cnoisy, cfiltered))
cv2.imshow('Original, Noisy, Gaussian-filtered', cframe)

# 'q' 키를 누르면 종료
key = cv2.waitKey(0)  # 대기 시간을 무한대로 설정하여 사용자가 키를 누를 때까지 기다립니다.
if key == ord('q'):
    cv2.destroyAllWindows()
