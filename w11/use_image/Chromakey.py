import cv2
import numpy as np

W = 320
H = 240

# 배경 이미지 로드
img_path = 'D:/code/mmp/w11/background2.jpg'
bimg = cv2.imread(img_path)
bimg = cv2.resize(bimg, (W, H))
bb, bg, br = cv2.split(bimg)

# 처리할 이미지 경로 입력받기
img_path2 = 'D:/code/mmp/w11/equal3.jpg'


while True:
    # 경로에서 이미지 읽기
    img = cv2.imread(img_path2)
    img = cv2.resize(img, (W, H))
    
    if img is not None:
        b, g, r = cv2.split(img)
        for y in range(H):
            for x in range(W):
                if g.item(y,x) <50 and b.item(y,x) <50 and r.item(y,x) < 50:
                #if (g.item(y, x) > 50 and g.item(y, x) > r.item(y, x)
                #     + 30 and g.item(y, x) > b.item(y, x) + 30):

                    b.itemset(y, x, bb.item(y,x))
                    g.itemset(y, x, bg.item(y,x))
                    r.itemset(y, x, br.item(y,x))
        timg = cv2.merge((b, g, r))
        cimg = np.hstack((bimg, timg))
        cv2.imshow('Chroma key', cimg)
        key = cv2.waitKey(33)
        if key == ord('q'):
            break

cv2.destroyAllWindows()
