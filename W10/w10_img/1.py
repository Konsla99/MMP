import cv2
import numpy as np

# 이미지 경로 설정
print("Select an image to process:")
print("1: circle  Image")
print("2: Many circle  Image")
print("3: x line  Image")
print("4: ordered  Image")
print("5: many line Image")
print("6: not linear  Image")
print("7: not linear ordered Image")
print("8: star Image")
choice = input("Enter your choice (1 ~ 8): ")

if choice == '1':
    image_path = 'D:/code/mmp/W10/w10_img/f_domain/real.jpg'
elif choice == '2':
    image_path = 'D:/code/mmp/W10/w10_img/f_domain/real2.jpg'
elif choice == '3':
    image_path = 'D:/code/mmp/W10/w10_img/f_domain/low.jpg'
elif choice == '4':
    image_path = 'D:/code/mmp/W10/w10_img/f_domain/ordered_line.jpg'
elif choice == '5':
    image_path = 'D:/code/mmp/W10/w10_img/f_domain/many_line.jpg'    
elif choice == '6':
    image_path = 'D:/code/mmp/W10/w10_img/f_domain/not_linear.jpg'
elif choice == '7':
    image_path = 'D:/code/mmp/W10/w10_img/f_domain/not_linear_ordered.jpg'   
elif choice == '8':
    image_path = 'D:/code/mmp/W10/w10_img/f_domain/star.jpg'
elif choice == '9':
    image_path = 'D:/code/mmp/W10/w10_img/high2.jpg'
else:
    print("Invalid choice. Exiting...")
    exit()

# 이미지 읽기
img = cv2.imread(image_path)

width = 480  # 예시 너비2
height = 320  # 예시 높이

# 이미지를 지정된 크기로 조정
img = cv2.resize(img, (width, height))

# 그레이스케일 이미지 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2차원 푸리에 변환
F = np.fft.fft2(gray)
Fshift = np.fft.fftshift(F)
mag_spc = np.clip(20 * np.log(np.abs(Fshift)), 0, 255).astype(np.uint8)

# 원본 그레이스케일 이미지와 매그니튜드 스펙트럼을 가로로 합치기
cframe = np.hstack((gray, mag_spc))

# 결과 이미지 보여주기
cv2.imshow('2D-FFT', cframe)
cv2.waitKey(0)  # 사용자가 키를 누를 때까지 기다림
cv2.destroyAllWindows()
