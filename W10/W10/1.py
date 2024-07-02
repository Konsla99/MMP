import cv2
import numpy as np

# 비디오 파일 경로 설정
video_path = 'D:/code/mmp/W10/test.mp4'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)
width = 640  # 새 너비
height = 480  # 새 높이
while True:
    ret, img = cap.read()  # 프레임 읽기
    if ret:
        img = cv2.resize(img, (width, height))

        # 그레이스케일 이미지 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2차원 푸리에 변환
        F = np.fft.fft2(gray)
        # 변환된 결과에서 DC 성분(저주파)을 이미지 중앙으로 이동
        Fshift = np.fft.fftshift(F)
        # 로그 스케일 변환과 클리핑을 통해 변환 결과의 절대값을 로그 스케일로 변환하고, 클리핑해 8비트 이미지로 수정
        mag_spc = np.clip(20 * np.log(np.abs(Fshift)), 0, 255).astype(np.uint8)

        # 원본 그레이스케일 이미지와 매그니튜드 스펙트럼을 가로로 합치기
        cframe = np.hstack((gray, mag_spc))

        # 결과 이미지 보여주기
        cv2.imshow('2D-FFT', cframe)

        # 'q'를 누를 때까지 30ms 간격으로 대기
        key = cv2.waitKey(33)
        if key == ord('q'):
            break
    else:
        break  # 프레임 읽기 실패 시 루프 종료

cap.release()  # 캡처 객체 해제
cv2.destroyAllWindows()  # 모든 창 닫기
