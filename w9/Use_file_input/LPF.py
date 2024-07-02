# 해상도 늘려보기 cap.set부분
# cv2. filter2d의  bordertype 변경해보기
# default = border_reflect_101 
# 이건 LPF에서 가장 성능이 좋다
# Border_constant 는 HPF에서 좋다 그러므로 이거 해보자
# BOrder_Replicate는 임의의 값을 패딩 한다 총 3개의 시도
import cv2
import numpy as np

ksize = int(input('Enter kernel size:'))
kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)

# 이미지 파일 경로 직접 설정
image_path = 'D:/code/mmp/w9/data/test.jpg'  # 이미지 경로를 여기에 입력하세요

# 이미지 파일 읽기
frame = cv2.imread(image_path)
if frame is None:
    print("Error: Image could not be read. Check the path.")
    exit()

# 이미지 해상도 설정을 유지
width = 640
height = 480
frame = cv2.resize(frame, (width, height))  # 이미지를 지정된 width와 height로 조정

# 이미지에 필터 적용
filtered_frame = cv2.filter2D(frame, -1, kernel)

# 원본 이미지와 필터링된 이미지를 가로로 나란히 붙여서 보여줌
cframe = np.hstack((frame, filtered_frame))
cv2.imshow('Image', cframe)

# 'q' 키를 누르면 종료
key = cv2.waitKey(0)  # 대기 시간을 무한대로 설정하여 사용자가 키를 누를 때까지 기다립니다.
if key == ord('q'):
    cv2.destroyAllWindows()
