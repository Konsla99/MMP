import cv2
import numpy as np

N = 256
print("Select an image to process:")
print("1: Low Frequency Image")
print("2: Mid Frequency Image")
print("3: High Frequency Image")
choice = input("Enter your choice (1, 2, or 3): ")

if choice == '1':
    image_path = 'D:/code/mmp/W10/w10_img/low.jpg'
elif choice == '2':
    image_path = 'D:/code/mmp/W10/w10_img/mid.jpg'
elif choice == '3':
    image_path = 'D:/code/mmp/W10/w10_img/high.jpg'
else:
    print("Invalid choice. Exiting...")
    exit()

# 이미지 파일 읽기
img = cv2.imread(image_path)

if img is not None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 256
    r_gray = cv2.resize(gray, (N, N))
    F = np.fft.rfft2(r_gray)
    F_sh = np.fft.fftshift(F)
    iF_sh = np.fft.ifftshift(F_sh)
    i_gray = np.clip(np.abs(np.fft.irfft2(iF_sh)), 0, 255)
    cframe = np.hstack((r_gray, i_gray))
    cv2.imshow('2D-FFT', cframe)
    cv2.waitKey(0)  # 키 입력 대기 시간 변경
else:
    print("Failed to load the image.")

cv2.destroyAllWindows()
