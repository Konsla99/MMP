
#pg 14 histogram

import cv2
from matplotlib import pyplot as plt
#RED,Blue,Green,Yellow,Purple
img_path= 'D:/code/mmp/w11/yellow.jpg'
# 이미지 로드
img = cv2.imread(img_path)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    img = cv2.resize(img, (320, 240))
cv2.imshow('img',img)    
plt.show()