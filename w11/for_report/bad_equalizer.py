import cv2
import numpy as np

# 사용자로부터 이미지 경로 입력받기
image_path = 'D:/code/mmp/w11/equal3.jpg'
img = cv2.imread(image_path)

if img is None:
    print("Image not found.")
else:
    img = cv2.resize(img, (720, 560))  # 이미지 크기 조정

    # BGR 채널로 이미지 분리
    b, g, r = cv2.split(img)

    # 각 채널에 대해 히스토그램 평활화 적용
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)

    # 평활화된 채널을 다시 합치기
    equalized_img = cv2.merge((b_eq, g_eq, r_eq))

    # 원본과 평활화된 이미지를 가로로 나란히 표시
    combined_img = np.hstack((img, equalized_img))

    # 결과 이미지 표시
    cv2.imshow('Original vs Equalized', combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
