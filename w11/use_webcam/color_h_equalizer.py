import cv2
import numpy as np

# 비디오 캡처 초기화
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)

while True:
    ret, img = cap.read()
    if ret:
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
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
