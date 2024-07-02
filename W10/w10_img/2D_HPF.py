import cv2
import numpy as np

def Build_H(N, f_radius):
    HN = N / 2
    H = np.ones((N, N), dtype=np.float32)  # 기본적으로 모든 값을 1로 설정
    for y in range(N):
        for x in range(N):
            fy = (y - HN) / HN
            fx = (x - HN) / HN
            radius = np.sqrt(fy**2 + fx**2)
            if radius <= f_radius:
                H[y][x] = 0  # 주어진 반경 내부는 0으로 설정
    return H

def Filter2D_FT(Fin, H):
    mag = np.abs(Fin)
    phs = np.angle(Fin)
    fmag = H * mag
    return fmag * np.exp(1j * phs)

# 이미지 파일 경로 설정
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
elif choice == '4':
    image_path = 'D:/code/mmp/W10/w10_img/qr.png'    
else:
    print("Invalid choice. Exiting...")
    exit()

# 이미지 파일 불러오기
img = cv2.imread(image_path)

N = 256
fcut = float(input("Enter cut off radius [0~1]: "))  # 사용자로부터 입력 받음
H = Build_H(N, fcut)
img_H = (H * 255).astype(np.uint8)  # 필터를 이미지로 시각화하기 위해 스케일 조정

if img is not None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255
    r_gray = cv2.resize(gray, (N, N))
    F = np.fft.fft2(r_gray)
    F_sh = np.fft.fftshift(F)
    FF = Filter2D_FT(F_sh, H)
    iF_sh = np.fft.ifftshift(FF)
    i_gray = np.clip(np.abs(np.fft.ifft2(iF_sh)), 0, 255)
    cframe = np.hstack((img_H, r_gray, i_gray))
    cv2.imshow('2D-FT High-Pass Filter Result', cframe)
    cv2.waitKey(0)  # 사용자가 키를 누를 때까지 기다림
    cv2.destroyAllWindows()
else:
    print("Failed to load the image.")
