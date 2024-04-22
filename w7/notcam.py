#cam 이 아닌 파일로부터 영상 읽기
import cv2

fname = input('Enter image file name:')
img  = cv2.imread(fname)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()