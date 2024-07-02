import cv2
import numpy as np

# 수정된 경로 (WSL Linux 형식)
img_path = '/mnt/d/code/mmp/W10/w10_img/mid2.jpg'
# 사용자로부터 이미지 경로 입력받기
image_path = 'D:/code/mmp/W10/w10_img/mid.jpg'

# 이미지를 읽고 크기 조정
img = cv2.imread(img_path)
img = cv2.resize(img, (320, 240))

while True:
    if img is not None:
        # BGR 이미지를 YCrCb 컬러 스페이스로 변환
        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        
        # YCrCb 컬러 스페이스에서 Y 채널(밝기)만 평활화
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        
        # 평활화된 Y 채널을 가진 이미지를 다시 BGR로 변환
        equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        
        # 원본과 평활화된 이미지를 가로로 나란히 표시 (side by side)
        combined_img = np.hstack((img, equalized_img))
        
        # 이미지 윈도우에 결과 표시
        cv2.imshow('Original vs Equalized', combined_img)
        
        # 'q'를 누르면 루프 종료
        key = cv2.waitKey(33)
        if key == ord('q'):
            break

cv2.destroyAllWindows()
