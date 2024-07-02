#평균 필터의 노이즈 제거원리는 무엇인가?
# 조사해볼것!!
# gain 도 변경해보자
# kernel도 극으로 변경

import cv2
import numpy as np

ksize = int(input('Enter kernel size:'))
kernel = np.ones((ksize, ksize), np.float32)/(ksize*ksize)
ngain = float(input('Enter noise gain:'))

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

# 이미지를 흑백으로 변환
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 노이즈 추가
noisy = np.clip(img + np.random.random((height, width)) * ngain, 0, 255).astype(np.uint8)

# 평균 필터 적용
filtered_img = cv2.filter2D(noisy, -1, kernel)

# 이미지들을 가로로 나란히 붙여서 보여줌
cframe = np.hstack((img, noisy, filtered_img))
cv2.imshow('Original, Noisy, Filtered', cframe)

# 'q' 키를 누르면 종료
while True:
    key = cv2.waitKey(33)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
