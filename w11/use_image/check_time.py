import numpy as np
import cv2
import time

# 감마 보정 방법 1: 직접 접근
def gammaCorrect1(im, gamma):
    outImg = np.zeros(im.shape, im.dtype)
    rows, cols = im.shape
    for i in range(rows):
        for j in range(cols):
            outImg[i][j] = ((im[i][j] / 255.0) ** (1 / gamma)) * 255
    return outImg

# 감마 보정 방법 2: item을 이용한 접근
def gammaCorrect2(im, gamma):
    outImg = np.zeros(im.shape, im.dtype)
    rows, cols = im.shape
    for i in range(rows):
        for j in range(cols):
            gammaValue = ((im.item(i, j) / 255.0) ** (1 / gamma)) * 255
            outImg.itemset(i, j, gammaValue)
    return outImg

# 감마 보정 방법 3: Look-up table을 이용한 접근
def gammaCorrect3(im, gamma):
    outImg = np.zeros(im.shape, im.dtype)
    rows, cols = im.shape
    LUT = [((i / 255.0) ** gamma) * 255 for i in range(256)]
    for i in range(rows):
        for j in range(cols):
            gammaValue = LUT[im.item(i, j)]
            outImg.itemset(i, j, gammaValue)
    return outImg

# 감마 보정 방법 4: Look-up table과 numpy 배열을 이용한 방법
def gammaCorrect4(im, gamma):
    LUT = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    outImg = LUT[im]
    return outImg

# 이미지 경로 입력받기 및 처리
image_path = 'D:/code/mmp/W10/w10_img/mid2.jpg'
img = cv2.imread(image_path)
if img is None:
    print("Image not found.")
else:
    img = cv2.resize(img, (480, 320))  # Resize the image to 480x320
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gamma = 0.1

    # 감마 보정 적용
    img1 = gammaCorrect1(gray, gamma)
    img2 = gammaCorrect2(gray, gamma)
    img3 = gammaCorrect3(gray, gamma)
    img4 = gammaCorrect4(gray, gamma)

    # 이미지를 두 행으로 나누어 표시
    top_row = np.hstack((gray, img1, img2))  # 첫 번째 행에 세 개의 이미지
    bottom_row = np.hstack((img3, img4))     # 두 번째 행에 두 개의 이미지

    # 첫 번째 행과 두 번째 행이 동일한 너비를 갖도록 조정
    # 두 번째 행에 빈 공간 추가 (검정색 이미지로 채움)
    if top_row.shape[1] != bottom_row.shape[1]:
        empty_img = np.zeros_like(img1)
        bottom_row = np.hstack((bottom_row, empty_img))

    # 두 행을 세로로 결합
    all_images = np.vstack((top_row, bottom_row))
    # 각각의 방법에 대한 계산 시간 측정
    start_time = time.time()
    gammaCorrect1(gray, gamma)
    print("Elapsed time for method 1: %.4f seconds" % (time.time() - start_time))

    start_time = time.time()
    gammaCorrect2(gray, gamma)
    print("Elapsed time for method 2: %.4f seconds" % (time.time() - start_time))

    start_time = time.time()
    gammaCorrect3(gray, gamma)
    print("Elapsed time for method 3: %.4f seconds" % (time.time() - start_time))

    start_time = time.time()
    gammaCorrect4(gray, gamma)
    print("Elapsed time for method 4: %.4f seconds" % (time.time() - start_time))

    cv2.imshow('Gamma Correction Comparison', all_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
