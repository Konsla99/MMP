import cv2
import numpy as np

N = 256
video_path = 'D:/code/mmp/W10/test.mp4'  # 동영상 파일 경로 설정

# 동영상 파일 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Video file cannot be opened or found.")
else:
    while True:
        ret, img = cap.read()
        if ret:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 256
            r_gray = cv2.resize(gray, (N, N))
            F = np.fft.rfft2(r_gray)
            F_sh = np.fft.fftshift(F)
            iF_sh = np.fft.ifftshift(F_sh)
            i_gray = np.clip(np.abs(np.fft.irfft2(iF_sh)), 0, 255)
            cframe = np.hstack((r_gray, i_gray))
            cv2.imshow('2D-FFT', cframe)

            key = cv2.waitKey(33)  # 30fps로 프레임 표시
            if key == ord('q'):  # 'q'를 누르면 루프 종료
                break
        else:
            break  # 프레임 읽기 실패 시 루프 종료

    cap.release()  # 캡처 객체 해제
    cv2.destroyAllWindows()  # 모든 창 닫기
