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

image_option = input("Enter option (1=2000, 2=1000, or 3=320x240): ")

if image_option == '1':
    image_path ='D:/code/mmp/w11/2000x2000.jpg'
elif image_option == '2':
    image_path ='D:/code/mmp/w11/1000x1000.jpg'
elif image_option == '3':
    image_path ='D:/code/mmp/w11/320x240.jpg'
else:
        print("Invalid option selected. Exiting.")
        exit()

img = cv2.imread(image_path)
if img is None:
    print("Image not found.")
else:
    print("Select image size:")
    print("1: 2000x2000")
    print("2: 1000x1000")
    print("3: 320x240")
    size_option = input("Enter option (1, 2, or 3): ")

    if size_option == '1':
        img = cv2.resize(img, (2000, 2000))
    elif size_option == '2':
        img = cv2.resize(img, (1000, 1000))
    elif size_option == '3':
        img = cv2.resize(img, (320, 240))
    else:
        print("Invalid option selected. Exiting.")
        exit()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gamma = 0.1
    #소수점 출력 늘려볼것
    start_time = time.time()
    img1 = gammaCorrect1(gray, gamma)
    print(f"Elapsed time for method 1: {time.time() - start_time:.8f} seconds")

    start_time = time.time()
    img2 = gammaCorrect2(gray, gamma)
    print(f"Elapsed time for method 2: {time.time() - start_time:.8f} seconds")

    start_time = time.time()
    img3 = gammaCorrect3(gray, gamma)
    print(f"Elapsed time for method 3: {time.time() - start_time:.8f} seconds")

    start_time = time.perf_counter()
    img4 = gammaCorrect4(gray, gamma)
    print(f"Elapsed time for method 4: {time.perf_counter() - start_time:.8f} seconds")
    start_time = time.perf_counter()
